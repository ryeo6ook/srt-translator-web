import streamlit as st
import os
import time
import logging
import io
import shutil
import tempfile
import queue
import threading
import requests
import json
from srt_translator import SRTTranslator, DEFAULT_MODELS, API_ENDPOINTS

# 预设文件路径
PRESETS_FILE = "presets.json"

def load_presets():
    if os.path.exists(PRESETS_FILE):
        try:
            with open(PRESETS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_presets(presets):
    with open(PRESETS_FILE, "w", encoding="utf-8") as f:
        json.dump(presets, f, ensure_ascii=False, indent=2)

# 配置页面
st.set_page_config(
    page_title="SRT 字幕翻译工具",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 使用队列处理日志，解决多线程 NoSessionContext 问题
class QueueLogHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        msg = self.format(record)
        self.log_queue.put(msg)

def fetch_models(api_endpoint, api_key):
    """从 OpenAI 兼容接口拉取模型列表"""
    if not api_endpoint:
        st.error("请先填写 API 端点")
        return None
    
    try:
        # 尝试推断 models 端点
        # 标准 OpenAI: https://api.openai.com/v1/chat/completions -> https://api.openai.com/v1/models
        # DeepSeek: https://api.deepseek.com/chat/completions -> https://api.deepseek.com/models
        
        base_url = api_endpoint
        if "/chat/completions" in base_url:
            base_url = base_url.replace("/chat/completions", "")
        if base_url.endswith("/"):
            base_url = base_url[:-1]
            
        url = f"{base_url}/models"
        
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            
        with st.spinner(f"正在从 {url} 拉取模型..."):
            response = requests.get(url, headers=headers, timeout=10)
            
        if response.status_code == 200:
            data = response.json()
            # OpenAI 格式: {"data": [{"id": "model-id", ...}, ...]}
            if "data" in data and isinstance(data["data"], list):
                return [model["id"] for model in data["data"] if "id" in model]
            else:
                st.warning("API 返回格式不符合 OpenAI 标准 (缺少 'data' 列表)")
                return None
        else:
            st.error(f"获取模型失败: {response.status_code} - {response.text[:100]}")
            return None
    except Exception as e:
        st.error(f"请求出错: {str(e)}")
        return None

def main():
    st.title("🎬 SRT 字幕翻译工具 (AI驱动)")
    st.markdown("""
    这是一个基于 AI 的 SRT 字幕翻译工具。上传您的字幕文件，配置 API，即可开始翻译。
    """)

    # 初始化 session state
    if "custom_models" not in st.session_state:
        st.session_state.custom_models = []
    if "user_prompt" not in st.session_state:
        st.session_state.user_prompt = ""

    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 参数配置")
        
        # API 配置
        st.subheader("API 设置")
        
        # 尝试从环境变量获取 Key (适配 Hugging Face Secrets)
        env_api_key = os.environ.get("API_KEY", "")
        
        api_type = st.selectbox("API 类型", ["deepseek", "grok", "custom"], index=0)
        
        api_key = st.text_input("API Key", value=env_api_key, type="password", help="如果您部署在 HF Spaces 并配置了 Secrets，这里会自动填充")
        
        if api_type == "custom":
            api_endpoint = st.text_input("API 端点 (Base URL)", value="https://api.deepseek.com/v1/chat/completions")
            
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🔄 拉取", help="尝试从 API 拉取可用模型列表"):
                    models = fetch_models(api_endpoint, api_key)
                    if models:
                        st.session_state.custom_models = models
                        st.success(f"获取到 {len(models)} 个模型")
            
            with col1:
                if st.session_state.custom_models:
                    model_name = st.selectbox("模型名称", st.session_state.custom_models, index=0)
                else:
                    model_name = st.text_input("模型名称", value="deepseek-chat")
        else:
            api_endpoint = None # 内部会自动处理
            default_model = DEFAULT_MODELS.get(api_type, "")
            model_name = st.text_input("模型名称", value=default_model)

        st.markdown("---")
        
        # 翻译参数
        st.subheader("翻译参数")
        temperature = st.slider("温度 (Temperature)", min_value=0.0, max_value=1.5, value=0.8, step=0.1, help="较高的值会使输出更随机，较低的值会更集中和确定。")
        batch_size = st.number_input("批次大小 (Batch Size)", min_value=1, max_value=50, value=5, help="每次发送给 AI 的字幕条数。")
        context_size = st.number_input("上下文大小 (Context Size)", min_value=0, max_value=10, value=2, help="提供给 AI 的前后文条数。")
        threads = st.number_input("线程数 (Threads)", min_value=1, max_value=10, value=1, help="并发请求数。注意：过高可能导致 API 限流。")
        
        st.markdown("---")
        
        # 高级选项
        st.subheader("高级选项")
        
        # 预设管理
        presets = load_presets()
        preset_names = ["-- 选择预设 --"] + list(presets.keys())
        
        def on_preset_change():
            if st.session_state.selected_preset != "-- 选择预设 --":
                st.session_state.user_prompt = presets[st.session_state.selected_preset]

        st.selectbox("加载预设提示词", preset_names, key="selected_preset", on_change=on_preset_change)
        
        user_prompt = st.text_area("用户提示词 (User Prompt)", height=100, key="user_prompt", placeholder="例如：请将所有专业术语翻译为简体中文...", help="您可以添加额外的指令来控制翻译风格。")
        
        with st.expander("💾 保存/管理预设"):
            new_preset_name = st.text_input("新预设名称", placeholder="例如：科技风格")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                if st.button("保存当前提示词"):
                    if new_preset_name and user_prompt:
                        presets[new_preset_name] = user_prompt
                        save_presets(presets)
                        st.success(f"已保存: {new_preset_name}")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("请输入名称和内容")
            
            with col_p2:
                if st.session_state.selected_preset != "-- 选择预设 --":
                    if st.button("删除当前选中"):
                        del presets[st.session_state.selected_preset]
                        save_presets(presets)
                        st.success("已删除")
                        time.sleep(1)
                        st.rerun()

        bilingual = st.checkbox("生成双语字幕", value=False)
        literal_align = st.checkbox("逐条逐句对齐 (直译优先)", value=False)
        structured_output = st.checkbox("结构化输出 (JSON)", value=False, help="尝试强制模型返回 JSON 格式，以保证行数对应。")
        professional_mode = st.checkbox("专业模式 (处理断句)", value=False)

    # 主区域
    uploaded_file = st.file_uploader("上传 SRT 文件", type=["srt"])

    if uploaded_file is not None:
        st.info(f"已上传: {uploaded_file.name}")
        
        if st.button("🚀 开始翻译", type="primary"):
            if not api_key:
                st.error("请先在侧边栏配置 API Key！")
                return

            # 创建临时文件保存上传的内容
            # 使用 tempfile.mkdtemp 创建一个临时目录，确保文件名不冲突
            temp_dir = tempfile.mkdtemp()
            input_path = os.path.join(temp_dir, uploaded_file.name)
            output_path = input_path + "_translated.srt"
            
            with open(input_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            # 进度显示区域
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_container = st.empty()
            
            # 初始化日志队列
            log_queue = queue.Queue()
            
            # 配置日志
            logger = logging.getLogger("SRT-Translator")
            logger.setLevel(logging.INFO)
            # 清除旧的 handlers 防止重复
            logger.handlers = []
            handler = QueueLogHandler(log_queue)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)

            # 翻译状态
            translation_status = {"success": False, "error": None}

            def run_translation():
                try:
                    # 动态更新模块级别的配置
                    if api_type == "custom" and api_endpoint:
                        API_ENDPOINTS["custom"] = api_endpoint
                        DEFAULT_MODELS["custom"] = model_name

                    translator = SRTTranslator(
                        api_type=api_type,
                        api_key=api_key,
                        batch_size=batch_size,
                        context_size=context_size,
                        max_workers=threads,
                        model_name=model_name,
                        user_prompt=user_prompt,
                        bilingual=bilingual,
                        temperature=temperature,
                        literal_align=literal_align,
                        structured_output=structured_output,
                        professional_mode=professional_mode
                    )
                    
                    translator.translate_srt_file(
                        input_file=input_path,
                        output_file=output_path,
                        resume=True 
                    )
                    translation_status["success"] = True
                except Exception as e:
                    translation_status["error"] = str(e)
                    logging.exception(e)

            # 在新线程中启动翻译
            t = threading.Thread(target=run_translation)
            t.start()

            status_text.text("正在翻译中，请稍候...")
            
            # 主循环：更新日志和进度
            log_content = ""
            while t.is_alive():
                # 消费队列中的所有日志
                while not log_queue.empty():
                    try:
                        msg = log_queue.get_nowait()
                        log_content += msg + "\n"
                    except queue.Empty:
                        break
                
                # 更新 UI
                if log_content:
                    display_lines = log_content.split('\n')[-20:]
                    log_container.code('\n'.join(display_lines), language='text')
                
                time.sleep(0.5)

            # 线程结束，最后刷新一次日志
            while not log_queue.empty():
                try:
                    msg = log_queue.get_nowait()
                    log_content += msg + "\n"
                except queue.Empty:
                    break
            
            if log_content:
                display_lines = log_content.split('\n')[-20:]
                log_container.code('\n'.join(display_lines), language='text')

            # 检查结果
            if translation_status["success"]:
                progress_bar.progress(100)
                status_text.success("翻译完成！")
                
                if os.path.exists(output_path):
                    with open(output_path, "r", encoding="utf-8") as f:
                        translated_content = f.read()
                    
                    output_filename = f"translated_{uploaded_file.name}"
                    st.download_button(
                        label="📥 下载翻译后的 SRT 文件",
                        data=translated_content,
                        file_name=output_filename,
                        mime="text/plain"
                    )
                else:
                    st.error("未找到输出文件，请检查日志。")
            else:
                st.error(f"翻译失败: {translation_status['error']}")

            # 清理临时文件 (可选，或者依赖系统自动清理)
            # 由于可能需要下载，这里暂时保留，或者等待一段时间
            # 在 Streamlit 中，如果不删除，临时文件可能会堆积。
            # 但如果在这里删除，用户点击下载按钮时可能文件已经不存在了？
            # 不，st.download_button 的 data 已经读取到内存中了 (translated_content)。
            # 所以可以安全删除。
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

if __name__ == "__main__":
    main()
