import os
import json
import base64
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import edge_tts

load_dotenv()

# --- 1. 定义状态 ---
class AgentState(TypedDict):
    query: str
    raw_search_results: List[Dict] # 存储原始搜索结果用于匹配 URL
    news_data: List[str]
    logs: List[str]
    final_report: str
    audio_b64: str

# --- 2. 初始化 ---
# 推荐使用 DeepSeek V3 (逻辑强且便宜) 或 GPT-4o
llm = ChatOpenAI(
    model="deepseek-chat", 
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.1 # 低温度保证引用准确
)

# 增加搜索数量，Tavily 一次最多 5-10 条，我们可能需要多次调用
search_tool = TavilySearchResults(max_results=5) 

# --- 3. 节点定义 ---

def search_node(state: AgentState):
    """宏观数据搜集员 (搜索 10+ 个源)"""
    logs = state.get("logs", [])
    logs.append("🌍 [Macro Scout] 正在启动全网宏观数据扫描...")
    
    # 定义两个维度的搜索词，确保覆盖面达到 10 个源
    search_queries = [
        "latest US GDP CPI inflation Fed interest rate data official",
        "China GDP PMI manufacturing exports imports data current month",
        "Global commodities gold oil bitcoin price trends today",
        "Major central banks policy rates and bond yields 10y"
    ]
    
    all_results = []
    seen_urls = set()
    
    for q in search_queries:
        try:
            logs.append(f"🔍 搜索维度: {q}...")
            results = search_tool.invoke(q)
            
            for res in results:
                if res['url'] not in seen_urls:
                    seen_urls.add(res['url'])
                    # 给每个内容打上 ID，方便 LLM 引用
                    all_results.append({
                        "id": len(all_results) + 1,
                        "url": res['url'],
                        "content": res['content'],
                        "title": res['url'] # 简化标题
                    })
        except Exception as e:
            print(f"Search error: {e}")
            
    logs.append(f"✅ [Macro Scout] 共采集到 {len(all_results)} 个独立宏观数据源。")
    
    # 将结果格式化为文本喂给 LLM
    context_text = ""
    for item in all_results:
        context_text += f"Source_ID [{item['id']}] (URL: {item['url']}): {item['content']}\n\n"
        
    return {"raw_search_results": all_results, "news_data": [context_text], "logs": logs}

def analyst_node(state: AgentState):
    """首席宏观分析师 (严格格式控制)"""
    logs = state.get("logs", [])
    logs.append("🧠 [Chief Analyst] 正在进行数据交叉验证与合成计算...")
    
    context = state['news_data'][0]
    
    # 核心 Prompt：强制要求数字链接和公式展示
    prompt = ChatPromptTemplate.from_template("""
    你是一位华尔街顶级宏观对冲基金的首席策略师。请基于提供的【数据源列表】，撰写一份《全球宏观深度研报》。

    【严格约束】
    1. **引用即链接**：报告中出现的所有核心数据（如 GDP、CPI、利率、价格），必须做成 Markdown 链接格式，指向原始 URL。
       - 格式：`[数值](URL)`
       - 错误示范：GDP is 5.2% (Source 1)
       - 正确示范：US GDP grew by [5.2%](https://bea.gov/...)
    
    2. **公式展示**：如果你在报告中对数据进行了加工（如计算实际利率、价差、同比环比变化），必须在旁边用括号注明计算公式。
       - 格式：`[合成数据](URL) (计算公式: 名义利率 A - 通胀率 B)`
       - 例子：Real Yield is [2.1%](url1) (Formula: [10Y Yield 5.1%](url2) - [CPI 3.0%](url3))

    3. **数据源要求**：必须覆盖至少 5 个不同的宏观指标/来源。

    【报告结构】
    ## 🎯 核心摘要 (Key Takeaways)
    ## 🌏 全球宏观概览 (Global Macro)
    ## 💵 资产定价模型 (Valuation Models) -> 这里展示合成数据和公式
    ## 💡 交易策略 (Actionable Insights)
    ## 🔗 数据源列表 (Data Sources) -> 列出所有用到的 URL

    【数据源列表】:
    {context}
    """)
    
    chain = prompt | llm
    response = chain.invoke({"context": context})
    
    logs.append("🚀 [System] 深度研报构建完成。")
    return {"final_report": response.content, "logs": logs}

async def speech_node(state: AgentState):
    """语音合成 (仅朗读摘要，避免读 URL)"""
    logs = state.get("logs", [])
    # 简单截取前 500 字做语音，防止朗读 URL 体验不好
    text_to_read = state['final_report'][:500].replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    
    audio_b64 = ""
    try:
        communicate = edge_tts.Communicate(text_to_read, "zh-CN-YunxiNeural")
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
    except:
        pass
    
    return {"audio_b64": audio_b64}

# --- 4. 构建图 ---
workflow = StateGraph(AgentState)
workflow.add_node("macro_scout", search_node)
workflow.add_node("chief_analyst", analyst_node)
workflow.add_node("speech_synthesizer", speech_node)

workflow.set_entry_point("macro_scout")
workflow.add_edge("macro_scout", "chief_analyst")
workflow.add_edge("chief_analyst", "speech_synthesizer")
workflow.add_edge("speech_synthesizer", END)

app_graph = workflow.compile()

# --- 5. API ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ReportRequest(BaseModel):
    topic: str = "Macro"

@app.post("/generate_report")
async def generate_report(req: ReportRequest):
    inputs = {"query": req.topic, "logs": []}
    result = await app_graph.ainvoke(inputs)
    return {
        "report": result["final_report"],
        "logs": result["logs"],
        "audio": result["audio_b64"]
    }