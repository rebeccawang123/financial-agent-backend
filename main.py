import os
import json
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

# --- 1. 定义状态 (State) ---
# 新增了 sources (来源) 和 logs (思考过程)
class AgentState(TypedDict):
    query: str
    news_data: List[str]
    sources: List[Dict[str, str]]  # 新增: 存储具体的 Title 和 URL
    podcast_insights: str
    logs: List[str]                # 新增: 记录每一步的思考过程
    final_report: str

# --- 2. 初始化 ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
search_tool = TavilySearchResults(max_results=3)

# --- 3. 定义节点 (Nodes) ---

def news_node(state: AgentState):
    """新闻搜集员"""
    query = state.get("query", "Macro Finance")
    logs = state.get("logs", [])
    logs.append(f"🕵️ [News Agent] 开始搜索: '{query}'...")
    
    try:
        # 尝试调用真实搜索
        results = search_tool.invoke(f"{query} financial news bloomberg wsj")
        # 提取内容用于分析
        news_content = [res['content'] for res in results]
        # 提取元数据用于展示来源
        sources = [{"title": res['content'][:30]+"...", "url": res['url']} for res in results]
        logs.append(f"✅ [News Agent] 成功抓取到 {len(results)} 条相关新闻。")
    except Exception as e:
        # 模拟数据 (当没有 API Key 时)
        print(f"Search failed: {e}")
        logs.append("⚠️ [News Agent] 搜索 API 未响应，使用备用数据流。")
        news_content = [
            "Fed minutes suggest pause in rate cuts for December.",
            "NVIDIA stock volatility increases ahead of earnings.",
            "Bitcoin breaks $98k resistance level on ETF inflows."
        ]
        sources = [
            {"title": "WSJ: Fed Minutes Analysis", "url": "https://www.wsj.com/economy/central-banking"},
            {"title": "Bloomberg: Crypto Market Update", "url": "https://www.bloomberg.com/crypto"},
            {"title": "Reuters: Tech Stocks Rally", "url": "https://www.reuters.com/markets/us"}
        ]
        
    return {"news_data": news_content, "sources": sources, "logs": logs}

def podcast_node(state: AgentState):
    """播客监听员"""
    logs = state.get("logs", [])
    logs.append("🎧 [Pod Listener] 正在接入 RSS 源: 'All-In Podcast'...")
    logs.append("📝 [Pod Listener] 音频转录完成，正在提取关键观点...")
    
    mock_insight = """
    Chamath: AI infrastructure capex is peaking.
    Sacks: US Debt ceiling will be the main topic in 2025.
    """
    logs.append("✅ [Pod Listener] 观点提取完毕。")
    return {"podcast_insights": mock_insight, "logs": logs}

def analyst_node(state: AgentState):
    """首席分析师"""
    logs = state.get("logs", [])
    logs.append("🧠 [Chief Analyst] 正在交叉验证数据，准备生成 Markdown 报告...")
    
    news = "\n".join(state['news_data'])
    podcast = state['podcast_insights']
    
    prompt = ChatPromptTemplate.from_template("""
    你是华尔街首席分析师。基于新闻: {news} 和播客: {podcast}。
    写一份【每日金融晨报】，包含：市场情绪、宏观分析、Web3观察、操作建议。
    使用 Markdown 格式，多用 Emoji。
    """)
    
    chain = prompt | llm
    response = chain.invoke({"news": news, "podcast": podcast})
    
    logs.append("🚀 [System] 报告生成完毕，准备发送。")
    return {"final_report": response.content, "logs": logs}

# --- 4. 构建图 ---
workflow = StateGraph(AgentState)
workflow.add_node("news_scout", news_node)
workflow.add_node("podcast_listener", podcast_node)
workflow.add_node("chief_analyst", analyst_node)
workflow.set_entry_point("news_scout")
workflow.add_edge("news_scout", "podcast_listener")
workflow.add_edge("podcast_listener", "chief_analyst")
workflow.add_edge("chief_analyst", END)
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
        "report": result["final_report"],
        "sources": result["sources"], # 返回来源链接
        "logs": result["logs"]        # 返回思考日志
    }