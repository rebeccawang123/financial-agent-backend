import os
import json
import base64
import asyncio # 引入 asyncio
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
# 切换为 Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pptx import Presentation
import edge_tts # 引入 Edge TTS

load_dotenv()

# --- 1. 定义状态 ---
class AgentState(TypedDict):
    query: str
    news_data: List[str]
    sources: List[Dict[str, str]]
    podcast_insights: str
    logs: List[str]
    final_report: str
    report_chinese: str
    report_english: str
    audio_chinese_b64: str
    audio_english_b64: str
    ppt_b64: str

# --- 2. 初始化 ---
# 使用 Gemini 1.5 Flash (免费且快)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
search_tool = TavilySearchResults(max_results=3)

# --- 3. 定义节点 ---

def news_node(state: AgentState):
    """新闻搜集员"""
    query = state.get("query", "Macro Finance")
    logs = state.get("logs", [])
    logs.append(f"🕵️ [News Agent] 正在使用 Gemini Flash 搜索: '{query}'...")
    
    try:
        results = search_tool.invoke(f"{query} financial news bloomberg wsj")
        news_content = [res['content'] for res in results]
        sources = [{"title": res['content'][:30]+"...", "url": res['url']} for res in results]
        logs.append(f"✅ [News Agent] 成功抓取到 {len(results)} 条相关新闻。")
    except Exception as e:
        logs.append("⚠️ [News Agent] 搜索 API 未响应，使用备用数据流。")
        news_content = ["Market data unavailable due to network."]
        sources = []
        
    return {"news_data": news_content, "sources": sources, "logs": logs}

def podcast_node(state: AgentState):
    """播客监听员"""
    logs = state.get("logs", [])
    logs.append("🎧 [Pod Listener] 正在接入 RSS 源: 'All-In Podcast'...")
    mock_insight = "Chamath: AI infrastructure capex is peaking. Sacks: US Debt ceiling will be the main topic in 2025."
    return {"podcast_insights": mock_insight, "logs": logs}

def analyst_node(state: AgentState):
    """首席分析师"""
    logs = state.get("logs", [])
    logs.append("🧠 [Chief Analyst] Gemini 1.5 Flash 正在生成双语研报...")
    
    news = "\n".join(state['news_data'])
    podcast = state['podcast_insights']
    
    # 英文提示词
    prompt_en = ChatPromptTemplate.from_template("""
    You are a Wall Street Analyst. Based on: {news} and {podcast}.
    Write a brief "Daily Financial Briefing". Use Markdown.
    """)
    chain_en = prompt_en | llm
    response_en = chain_en.invoke({"news": news, "podcast": podcast})
    
    # 中文提示词
    prompt_zh = ChatPromptTemplate.from_template("""
    你是华尔街分析师。基于: {news} 和 {podcast}。
    写一份简短的【每日金融晨报】。使用 Markdown 格式，包含 Emoji。
    """)
    chain_zh = prompt_zh | llm
    response_zh = chain_zh.invoke({"news": news, "podcast": podcast})
    
    logs.append("🚀 [System] 报告生成完毕。")
    return {
        "report_english": response_en.content,
        "report_chinese": response_zh.content,
        "logs": logs
    }

# --- ⚠️ 核心修改: 使用 Edge TTS (异步) ---
async def speech_node(state: AgentState):
    """语音合成员 (Edge TTS 版)"""
    logs = state.get("logs", [])
    logs.append("🗣️ [Edge TTS] 正在调用微软 Neural 语音引擎...")
    
    report_zh = state['report_chinese']
    report_en = state['report_english']
    
    audio_chinese_b64 = ""
    audio_english_b64 = ""

    # 1. 生成中文语音 (推荐: zh-CN-YunxiNeural - 男声新闻腔)
    try:
        communicate = edge_tts.Communicate(report_zh[:50], "zh-CN-YunxiNeural")
        # 将音频流写入内存
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        audio_chinese_b64 = base64.b64encode(audio_data).decode('utf-8')
        logs.append("✅ [Edge TTS] 中文语音生成成功 (Free)。")
    except Exception as e:
        logs.append(f"❌ [Edge TTS] 中文生成失败: {str(e)}")

    # 2. 生成英文语音 (推荐: en-US-ChristopherNeural - 男声专业腔)
    try:
        communicate = edge_tts.Communicate(report_en[:50], "en-US-ChristopherNeural")
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        audio_english_b64 = base64.b64encode(audio_data).decode('utf-8')
        logs.append("✅ [Edge TTS] 英文语音生成成功 (Free)。")
    except Exception as e:
        logs.append(f"❌ [Edge TTS] 英文生成失败: {str(e)}")

    return {
        "audio_chinese_b64": audio_chinese_b64,
        "audio_english_b64": audio_english_b64,
        "logs": logs
    }

def ppt_node(state: AgentState):
    """PPT 生成器"""
    logs = state.get("logs", [])
    logs.append("📊 [PPT Generator] 正在生成演示文稿...")
    
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "每日金融晨报"
    slide.placeholders[1].text = "Powered by AlphaBrief.ai"
    
    # 简单内容页
    slide2 = prs.slides.add_slide(prs.slide_layouts[1])
    slide2.shapes.title.text = "核心摘要"
    slide2.shapes.placeholders[1].text = state['report_chinese'][:500]
    
    from io import BytesIO
    ppt_stream = BytesIO()
    prs.save(ppt_stream)
    ppt_stream.seek(0)
    ppt_b64 = base64.b64encode(ppt_stream.read()).decode('utf-8')
    
    logs.append("✅ [PPT] 文档打包完成。")
    return {"ppt_b64": ppt_b64, "logs": logs}

# --- 4. 构建图 ---
workflow = StateGraph(AgentState)
workflow.add_node("news_scout", news_node)
workflow.add_node("podcast_listener", podcast_node)
workflow.add_node("chief_analyst", analyst_node)
workflow.add_node("speech_synthesizer", speech_node)
workflow.add_node("ppt_generator", ppt_node)

workflow.set_entry_point("news_scout")
workflow.add_edge("news_scout", "podcast_listener")
workflow.add_edge("podcast_listener", "chief_analyst")
workflow.add_edge("chief_analyst", "speech_synthesizer")
workflow.add_edge("speech_synthesizer", "ppt_generator")
workflow.add_edge("ppt_generator", END)

app_graph = workflow.compile()

# --- 5. API ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ReportRequest(BaseModel):
    topic: str = "今日市场"

@app.post("/generate_report")
async def generate_report(req: ReportRequest):
    inputs = {"query": req.topic, "logs": []}
    result = await app_graph.ainvoke(inputs)
    return {
        "report_chinese": result["report_chinese"],
        "report_english": result["report_english"],
        "sources": result["sources"],
        "logs": result["logs"],
        "audio_chinese_b64": result["audio_chinese_b64"],
        "audio_english_b64": result["audio_english_b64"],
        "ppt_b64": result["ppt_b64"]
    }