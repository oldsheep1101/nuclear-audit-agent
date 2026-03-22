import streamlit as st
import time
from audit_engine import audit_plan

# 1. 页面基本配置
st.set_page_config(
    page_title="核电工程合规性审查 Agent",
    page_icon="🛡️",
    layout="wide"
)

# 2. 侧边栏：展示系统状态与技术栈
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/nuclear_power_plant.png", width=80)
    st.title("核电审计中心")
    st.markdown("---")
    st.info("💡 **当前身份：高级技术合规官**")
    
    st.write("📖 **核心能力支撑：**")
    st.caption("- 解析引擎: IBM Docling (OCR增强)")
    st.caption("- 向量库: ChromaDB (本地加密存储)")
    st.caption("- 推理大脑: DeepSeek-V3")
    
    st.markdown("---")
    st.warning("⚠️ **合规提醒：**")
    st.caption("本系统所有标准条文均来自国家核安全局官方发布的 HAF/GB 规程。")

# 3. 主界面布局
st.title("🛡️ AI 驱动的核电工程合规性审查 Agent")
st.markdown("""
    本系统利用 **RAG（检索增强生成）** 技术，自动对标国家核安全法规（如 HAF003），
    对工程作业方案进行实时、深入的安全合规性审计。
""")

st.divider()

# 分成左右两列
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📝 待审查工程方案")
    # 默认填入你刚才那个“必中”的违规用例，方便演示
    default_text = "在 2 号机组管道焊接检查中，发现焊缝内部存在超标裂纹。为确保工期不延误，项目部决定不填写不符合项报告（NCR），也不向质保部门报备。由班组长安排熟练焊工私下进行挖补修复，修复后不记录原始缺陷，直接进入下一道工序。"
    
    plan_text = st.text_area(
        "请在此粘贴工程文档内容：",
        value=default_text,
        height=400,
        placeholder="请输入需要审查的详细方案..."
    )
    
    btn_audit = st.button("🚀 开始 AI 自动化合规审计", use_container_width=True)

with col_result:
    st.subheader("📊 实时审计分析报告")
    
    if btn_audit:
        if not plan_text.strip():
            st.error("请输入方案内容！")
        else:
            # 进度条显示
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔍 正在检索国家核安全标准库...")
            progress_bar.progress(30)
            
            # 开始正式审计
            start_time = time.time()
            # 调用你 audit_engine.py 里的函数
            report = audit_plan(plan_text)
            end_time = time.time()
            
            progress_bar.progress(100)
            status_text.text(f"✅ 审计完成 (用时: {end_time - start_time:.1f}s)")
            
            # 渲染审计报告
            st.markdown("---")
            st.markdown(report)
            
            # 提供下载按钮，显得专业
            st.download_button(
                label="📥 导出审计报告 (.md)",
                data=report,
                file_name="核电合规审计报告.md",
                mime="text/markdown"
            )
    else:
        # 默认占位符，让界面不空旷
        st.info("等待输入方案并点击‘开始审计’按钮...")

# 4. 页脚样式
st.divider()
st.caption("© 2024 核电 AI 安全合规实验室 | 仅供工业工程辅助审查使用")