import os
from typing import TypedDict, List
from dotenv import load_dotenv

# LangGraph & LangChain imports
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 加载环境变量 (.env 文件需包含 OPENAI_API_KEY 和 TAVILY_API_KEY)
load_dotenv()

# --- 1. 定义状态 (State) ---
# 这是智能体之间传递的“记忆包”
class AgentState(TypedDict):
    query: str              # 用户输入的初始意图
    news_data: List[str]    # 搜集到的新闻
    podcast_insights: str   # 播客摘要
    final_report: str       # 最终生成的 Markdown 报告

# --- 2. 初始化工具和模型 ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # 或使用 Claude-3-5-sonnet
search_tool = TavilySearchResults(max_results=3) # 强大的搜索工具

# --- 3. 定义节点 (Nodes / Agents) ---

def news_node(state: AgentState):
    """新闻搜集员: 负责搜索最新的金融新闻"""
    print("--- 🕵️ News Agent Working ---")
    query = state.get("query", "今日宏观市场分析")
    
    # 真实场景调用搜索工具
    try:
        results = search_tool.invoke(f"{query} financial news bloomberg wsj")
        news_content = [res['content'] for res in results]
    except Exception:
        # 如果没有 API Key，回退到模拟数据，方便您调试
        news_content = [
            "美联储会议纪要暗示12月可能暂停降息。",
            "英伟达财报前夕股价波动加剧，期权市场看涨。",
            "比特币突破98k美元，ETF资金持续流入。"
        ]
        
    return {"news_data": news_content}

def podcast_node(state: AgentState):
    """播客监听员: 模拟分析热门播客"""
    print("--- 🎧 Podcast Agent Working ---")
    
    # 真实场景这里会调用 Whisper API 转录音频
    # 这里我们模拟“All-In Podcast”的摘要
    mock_insight = """
    在最新的 All-In Podcast 中，Chamath 提到 AI 基础设施投资周期可能接近尾声，
    资金将流向应用层。Sacks 认为美国债务问题将在 2025 年成为核心议题。
    """
    return {"podcast_insights": mock_insight}

def analyst_node(state: AgentState):
    """首席分析师: 汇总信息并写报告"""
    print("--- 🧠 Chief Analyst Working ---")
    
    news = "\n".join(state['news_data'])
    podcast = state['podcast_insights']
    
    prompt = ChatPromptTemplate.from_template("""
    你是一位华尔街资深分析师。请根据以下信息，写一份Markdown格式的【每日金融晨报】。
    
    【最新新闻】:
    {news}
    
    【播客观点】:
    {podcast}
    
    要求：
    1. 包含“市场情绪”、“宏观分析”、“Web3观察”和“操作建议”四个板块。
    2. 风格专业、犀利、简洁。
    3. 使用Emoji增加可读性。
    """)
    
    chain = prompt | llm
    response = chain.invoke({"news": news, "podcast": podcast})
    
    return {"final_report": response.content}

# --- 4. 构建图 (Graph Construction) ---
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("news_scout", news_node)
workflow.add_node("podcast_listener", podcast_node)
workflow.add_node("chief_analyst", analyst_node)

# 定义边 (执行顺序)
workflow.set_entry_point("news_scout")
workflow.add_edge("news_scout", "podcast_listener")
workflow.add_edge("podcast_listener", "chief_analyst")
workflow.add_edge("chief_analyst", END)

# 编译图
app_graph = workflow.compile()

# --- 5. FastAPI 部署接口 ---
app = FastAPI(title="Financial Agent API")

# 允许跨域 (让 React 前端能访问)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReportRequest(BaseModel):
    topic: str = "今日市场动态"

@app.post("/generate_report")
async def generate_report(req: ReportRequest):
    """前端调用的主接口"""
    inputs = {"query": req.topic, "news_data": [], "podcast_insights": "", "final_report": ""}
    
    # 调用 LangGraph 执行工作流
    result = await app_graph.ainvoke(inputs)
    
    return {
        "status": "success",
        "report": result["final_report"],
        "steps": ["News Scout", "Podcast Listener", "Chief Analyst"] # 用于前端显示进度
    }

# 运行方式: uvicorn main:app --reload