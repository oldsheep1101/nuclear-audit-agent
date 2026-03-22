import os
import shutil
import time
from typing import Optional

from dotenv import load_dotenv

# 1. 环境变量与镜像配置 (必须在任何 huggingface / transformers 导入之前)
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
# 大文件下载中途断流时放宽超时，便于镜像站重试续传
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings # 如果报错，请 pip install langchain-huggingface
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# --- 核心引入：Docling 解析引擎 ---
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import RapidOcrOptions, ThreadedPdfPipelineOptions

# --- 显式指定 OCR 配置类：优先尝试旧包路径，不存在则用 datamodel（当前版本仅有后者）---
try:
    from docling.ocr.rapid_ocr_engine import (  # type: ignore[import-not-found]
        RapidOcrOptions as RapidOcrOptionsEngine,
    )
except ImportError:
    RapidOcrOptionsEngine = RapidOcrOptions

# 配置常量
DEFAULT_PDF = "data/nuclear_standard.pdf"
DB_PATH = "./.chroma_db"
# 与向量库绑定的嵌入模型名（若更换模型，必须删 .chroma_db 后 force_rebuild=True 重建）
# 中文标准/方案检索用 bge-small-zh；体积约百兆级，比 multilingual-MiniLM-L12（~470MB）更不易下崩
EMBEDDING_MODEL_ID = "BAAI/bge-small-zh-v1.5"
EMBEDDING_MARKER = os.path.join(DB_PATH, ".embedding_model_id")
RETRIEVER_K = 8

_hf_embed: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """懒加载 + 重试：镜像站大文件下载常出现 IncompleteRead / ChunkedEncodingError。"""
    global _hf_embed
    if _hf_embed is not None:
        return _hf_embed

    from requests.exceptions import ChunkedEncodingError, ConnectionError
    from urllib3.exceptions import IncompleteRead, ProtocolError

    last_err: BaseException | None = None
    for attempt in range(1, 6):
        try:
            _hf_embed = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_ID,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            return _hf_embed
        except (
            ChunkedEncodingError,
            ConnectionError,
            OSError,
            IncompleteRead,
            ProtocolError,
        ) as e:
            last_err = e
            wait = min(45, 8 * attempt)
            print(
                f"--- ⚠️ 嵌入模型下载/加载失败（{attempt}/5）: {type(e).__name__}: {e}\n"
                f"    {wait} 秒后重试。若仍失败，请删除缓存中不完整的模型目录后再运行：\n"
                f"    ~/.cache/huggingface/hub/models--{EMBEDDING_MODEL_ID.replace('/', '--')} ---"
            )
            time.sleep(wait)
    raise RuntimeError(
        "多次重试后仍无法下载/加载嵌入模型。请检查网络、换时段重试，或清理 HuggingFace 缓存后再次运行。"
    ) from last_err


# 2. 初始化大模型 (DeepSeek)
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0, # 严谨模式
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)


def _retriever_from_store(vs: Chroma):
    return vs.as_retriever(search_kwargs={"k": RETRIEVER_K})


def _embedding_marker_ok() -> bool:
    if not os.path.isfile(EMBEDDING_MARKER):
        return False
    try:
        with open(EMBEDDING_MARKER, encoding="utf-8") as f:
            return f.read().strip() == EMBEDDING_MODEL_ID
    except OSError:
        return False


def _write_embedding_marker() -> None:
    os.makedirs(DB_PATH, exist_ok=True)
    with open(EMBEDDING_MARKER, "w", encoding="utf-8") as f:
        f.write(EMBEDDING_MODEL_ID)


def _cjk_ratio(text: str) -> float:
    """中文（CJK 统一汉字）占字符比例，用于判断解析结果是否像中文标准正文。"""
    if not text or not text.strip():
        return 0.0
    n = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return n / max(len(text), 1)


def _extract_text_pypdf(pdf_path: str) -> str:
    """直接从 PDF 文字层抽取（当 Docling 走布局/OCR 反而更差时作为对照）。"""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t and t.strip():
            parts.append(t)
    return "\n\n".join(parts)


def _document_converter_for_chinese_pdf() -> DocumentConverter:
    """很多中文标准 PDF 内嵌字形映射错误；整页 RapidOCR + 强制 CPU/限线程，避免无 GPU 环境报错。"""
    pipeline_options = ThreadedPdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.images_scale = 2.0

    ocr_options = RapidOcrOptionsEngine(
        lang=["ch_sim", "en"],
        force_full_page_ocr=True,
    )
    pipeline_options.ocr_options = ocr_options

    # 云端无 GPU 时避免走 auto/cuda；并限制线程，降低内存峰值
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=2,
        device=AcceleratorDevice.CPU,
    )

    # format_options 的值必须是 PdfFormatOption，不能直接传 pipeline_options
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        },
    )


def prepare_knowledge_base(pdf_path=DEFAULT_PDF, force_rebuild=False):
    """使用 Docling 解析 PDF 并构建向量库"""
    
    # 如果已经有数据库且不强制重建，直接加载
    if (
        os.path.exists(DB_PATH)
        and not force_rebuild
        and _embedding_marker_ok()
    ):
        print("--- 📦 发现现有数据库（嵌入模型一致），正在直接加载... ---")
        return _retriever_from_store(
            Chroma(persist_directory=DB_PATH, embedding_function=get_embeddings())
        )

    if os.path.exists(DB_PATH) and not force_rebuild and not _embedding_marker_ok():
        print(
            "--- ⚠️ 向量库与当前嵌入模型不一致（或缺少标记文件），将自动重建索引。 "
            "若你刚换过 EMBEDDING_MODEL_ID，这是正常现象。---"
        )
        force_rebuild = True

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ 找不到 PDF 文件: {pdf_path}")

    print(f"--- 🚀 正在启动 Docling 深度解析 (支持扫描件): {pdf_path} ---")
    print("--- 提示：解析扫描件较耗时，请耐心等待... ---")
    
    # A. 使用 Docling（强制整页中文 OCR，避免误用损坏的内嵌文字层）
    converter = _document_converter_for_chinese_pdf()
    result = converter.convert(pdf_path)
    
    # B. 导出为 Markdown (能极好地保留表格和层级结构)
    full_text = result.document.export_to_markdown()
    pypdf_text = _extract_text_pypdf(pdf_path)
    r_doc, r_pdf = _cjk_ratio(full_text), _cjk_ratio(pypdf_text)

    # 少数 PDF 内嵌层正常但 OCR 反而差：自动选中文占比更高的一侧
    if r_pdf > r_doc + 0.03 and r_pdf >= 0.06:
        print(
            f"--- ⚠️ Docling 输出中文占比 {r_doc:.1%} 低于 pypdf 内嵌文字 {r_pdf:.1%}，已改用 pypdf 文本建库。---"
        )
        full_text = pypdf_text
    elif r_doc < 0.05 and r_pdf < 0.05:
        print(
            "--- ⚠️ 中文占比极低（Docling 与 pypdf 均如此）。"
            "常见原因：扫描件 OCR 未就绪、或 PDF 为加密/纯图且分辨率过低。请检查终端是否有 RapidOCR 报错。---"
        )
    elif r_doc < 0.08:
        print(
            f"--- ⚠️ 当前正文中文占比约 {r_doc:.1%}，可能仍有乱码或 OCR 不理想；"
            "若预览仍非中文，请换「文字可选中」的 PDF 或提高扫描分辨率后重建索引。---"
        )

    if not full_text.strip():
        raise ValueError("❌ 解析失败：未能从 PDF 中提取出任何文字内容。")

    preview = full_text[:80].replace("\n", " ")
    print(f"--- ✅ 解析完成。中文占比约 {_cjk_ratio(full_text):.1%}，预览: {preview}... ---")

    # C. 文本切片（加入中文标点切分，避免整段过长块）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    splits = text_splitter.split_text(full_text)

    # D. 存入向量数据库
    print("--- 存储至本地向量数据库... ---")
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH) # 清理旧数据
        
    vectorstore = Chroma.from_texts(
        texts=splits,
        embedding=get_embeddings(),
        persist_directory=DB_PATH
    )
    _write_embedding_marker()
    return _retriever_from_store(vectorstore)


def format_context_docs(docs):
    """把检索到的片段显式编号，便于模型引用『具体条文』。"""
    if not docs:
        return "（检索未返回任何片段；请检查向量库是否已用当前嵌入模型重建、PDF 是否成功解析出中文正文。）"
    parts = []
    for i, d in enumerate(docs, start=1):
        parts.append(f"### 摘录 {i}\n{d.page_content.strip()}")
    return "\n\n".join(parts)

def audit_plan(plan_text):
    """核电合规审查核心函数"""
    try:
        retriever = prepare_knowledge_base()
        
    # --- 新增调试代码：直接看检索结果 ---
        print("\n🔍 [DEBUG] 正在从数据库检索条文...")
        search_docs = retriever.invoke(plan_text) # 模拟检索动作
    
        if not search_docs:
            print("❌ [DEBUG] 检索结果为空！数据库里找不到相关内容。")
        else:
            print(f"✅ [DEBUG] 检索到 {len(search_docs)} 条相关片段：")
            for i, doc in enumerate(search_docs):
                print(f"--- 片段 {i+1} ---\n{doc.page_content[:200]}...\n")
    # -------------------------------
        


        # 专业审计 Prompt
        template = """你是一位国家核安全局资深审查员。请严格根据提供的【核电安全标准条文】，对比【工程作业方案】，指出其中的违规项。
        
        【参考标准条文】：
        {context}
        
        【待审查方案】：
        {question}
        
        请按以下格式输出：
        ---
        ### 🛡️ 审查结论：[合格 / 存在风险]
        ### 1. 🟢 合规项：列出符合标准的操作。
        ### 2. 🔴 违规项：明确指出违反的具体条文及潜在安全隐患（如辐射超标、设备损坏）。
        ### 3. 📝 整改建议：给出具体的修改动作。
        ---
        若【参考标准条文】中确实没有任何与方案主题相关的摘录，再回答“参考库未覆盖此内容，建议转人工核验”。不要仅凭措辞不同就否认已给出的摘录。
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # 构建 RAG 链：必须先把 Document 列表格式化成字符串，否则模型看到的 context 结构不理想
        chain = (
            {
                "context": retriever | RunnableLambda(format_context_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )

        print("--- 🤖 AI 正在对比审查中... ---")
        response = chain.invoke(plan_text)
        return response.content
        
    except Exception as e:
        return f"❌ 运行出错: {str(e)}"

if __name__ == "__main__":
    # --- 测试用例：你可以修改这里的文字进行不同场景的测试 ---
    test_case = "在 2 号机组管道焊接检查中，发现焊缝内部存在超标裂纹。为确保工期不延误，项目部决定不填写不符合项报告（NCR），也不向质保部门报备。由班组长安排熟练焊工私下进行挖补修复，修复后不记录原始缺陷，直接进入下一道工序。"
    
    print("\n" + "=="*20)
    print("☢️ 核电合规审查 Agent 启动")
    print("=="*20)
    
    report = audit_plan(test_case)
    
    print("\n" + ">> 最终审查报告 <<")
    print(report)
    try:
        _vs = Chroma(persist_directory=DB_PATH, embedding_function=get_embeddings())
        print(f"数据库中现有的条目总数: {_vs._collection.count()}")
    except Exception as _e:
        print(f"（无法读取向量库条目数: {_e}）")