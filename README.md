---
title: SRT Translator
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
---

# SRT Subtitle Translator (Web Version)

这是一个基于 AI 的 SRT 字幕翻译工具，Web 版本适配 Hugging Face Spaces。

## 功能

*   **AI 翻译**：支持 DeepSeek, Grok 等 API，也支持自定义 OpenAI 兼容接口。
*   **SRT 解析**：完美保留时间轴和字幕序号。
*   **批量处理**：自动分批发送给 AI，支持长字幕文件。
*   **实时预览**：在 Web 界面实时查看翻译日志和进度。
*   **双语字幕**：可选择生成双语对照字幕。

## 使用方法

1.  在左侧栏配置 API Key 和其他参数。
2.  上传 `.srt` 字幕文件。
3.  点击 "开始翻译"。
4.  等待翻译完成后，点击下载按钮获取结果。

## 部署说明

本项目已适配 Hugging Face Spaces (Streamlit SDK)。

如果您 Fork 本项目并部署到自己的 Space，建议在 Space Settings -> Secrets 中添加 `API_KEY` 环境变量，这样就不需要每次手动输入 API Key。
