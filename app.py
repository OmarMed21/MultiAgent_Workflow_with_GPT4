import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
import openai
from openai import OpenAI
import operator
import hashlib
import numpy as np
import os
import json

os.environ["OPENAI_API_KEY"] = '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'
client = OpenAI()

class VectorStore:
    json_db =  {}
    def __init__(self):
        self.store = {}
        self.curr_key = None

    def add(self, query_vector, code):
        key = self._vector_to_key(query_vector)
        self.store[key] = code

    def get_data_key(self):
        return self.store[str(self.curr_key)]

    def __getter__(self):
        return self.store 

    def search(self, query_vector):
        key = self._vector_to_key(query_vector)
        return self.store.get(key, None)

    def _vector_to_key(self, vector):
        self.curr_key = np.round(vector, decimals=2).tolist()
        return str(self.curr_key)

    def store_update_json_db(self, json_file_path: str = "db.json"):
        # print("db" + self.store[self.curr_key])
        VectorStore.json_db.update(self.store)
        try:
            with open(json_file_path, 'r') as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = VectorStore.json_db
        data.update(VectorStore.json_db)
        with open(json_file_path, 'w') as file:
            json.dump(data, file, indent=4)

        print("Query store in JSON Db file")
        
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    code: str
    data: str 
    execution_result: str
    user_feedback: str

workflow = StateGraph(AgentState)
vector_store = VectorStore()

def get_vector_representation(text):
    hash_object = hashlib.md5(text.encode())
    hex_dig = hash_object.hexdigest()
    vector = np.frombuffer(hex_dig.encode(), 'uint8')
    return vector

def retrieval_node(state: AgentState) -> AgentState:
    query = state['messages'][-1].content
    query_vector = get_vector_representation(query)
    result = vector_store.search(query_vector)
    
    if result:
        return {"messages": state['messages'] + [AIMessage(content=f"Found similar query. Using stored code.")],
                "code": result, 
                "data": state.get("data", ""), 
                "execution_result": state.get("execution_result", ""), 
                "user_feedback": state.get("user_feedback", "")} 
    else:
        return coding_node(state)  

def coding_node(state: AgentState) -> AgentState:
    data = state['data'] + " , write a python code that manupulate the data and show the  chart if needed also print the output in table ,use hapi fhir server baseURL is http://hapi.fhir.org/baseR4/ and return 100 object" 
    chat_completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "Generate Python code to create a chart from the following data."}, 
                  {"role": "user", "content": data}]
    )
    code = json.loads(chat_completion.json())
    code = code["choices"][0]["message"]["content"]
    # print(f"\nCode : {code}")
    return {"messages": state['messages'], "code": code}

def review_node(state: AgentState) -> AgentState:
    code = state['code']
    chat_completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "Review the following Python code for errors."}, 
                  {"role": "user", "content": code}]
    )
    review = json.loads(chat_completion.json())
    review = review["choices"][0]["message"]["content"]
    print("\n Review" + review)
    
    if "error" in review.lower():
        return {"messages": state['messages'] + [AIMessage(content="Errors found. Code sent back for revision.")]}
    else:
        return {"messages": state['messages'] + [AIMessage(content="Code reviewed. No errors found.")]}

def execute_code_in_secure_environment(code):
    try:
        safe_builtins = {
            '__import__': __import__,
            'print': print,
        }
        exec_globals = {}
        exec(code, {'__builtins__': safe_builtins}, exec_globals)
        return True, exec_globals
    except Exception as e:
        return False, f"Execution error: {str(e)}"

def executor_node(state: AgentState) -> AgentState:
    start_code = state['code'].find("```python") + len("```python\n") 
    end_code = state['code'].find("```", start_code)  
    code = state['code'][start_code:end_code].strip()

    print(f"code: {code}")
    execution_result, _ = execute_code_in_secure_environment(code)
    return {"messages": state['messages'], "execution_result": execution_result, "more":_}

def feedback_node(state: AgentState) -> AgentState:
    # print(f"\n\n\n\n\n FEEDBACK : {state['user_feedback']}")
    feedback = state['user_feedback']
    state['user_feedback'] = feedback
    
    # If the feedback is positive, save it in the vector store
    if feedback.lower() == 'positive':
        query = state['messages'][-1].content
        query_vector = get_vector_representation(query)
        vector_store.add(query_vector, state['code'])
        vector_store.store_update_json_db()
        print("Query stored successfully")
    
    return state

def should_continue(state: AgentState) -> str:
    feedback = state.get("user_feedback", "").lower()
    last_message_content = state["messages"][-1].content if state["messages"] else ""
    execution_result = state.get("execution_result", "")
    # execution_result = executor_node(state)['execution_result']
    # print(f"exec :{execution_result}")

    if "end" in feedback:
        return "end"
    if "No similar query found. Proceed to coding." in last_message_content or "Errors found. Code sent back for revision." in last_message_content:
        return "go_to_coding"
    elif "Found similar query" in last_message_content or "Code reviewed. No errors found." in last_message_content:
        return "go_to_execute"
    if execution_result:
        return "go_to_feedback"
    if "positive" in feedback:
        return "complete"
    else:
        print(f"Unexpected state encountered. State messages: {state['messages']}")
        return "reassess"

workflow.add_node("retrieval", retrieval_node)
workflow.add_node("coding", coding_node)
workflow.add_node("review", review_node)
workflow.add_node("execute", executor_node)
workflow.add_node("feedback", feedback_node)
workflow.set_entry_point("retrieval")

workflow.add_conditional_edges("retrieval", should_continue, {
    "go_to_coding": "coding", 
    "go_to_execute": "execute", 
    "go_to_feedback": "feedback", 
    "complete": END,
    "end": END,
    "reassess": "retrieval" 
})

workflow.add_conditional_edges("coding", should_continue, {
    "review": "review", 
    "execute": "execute", 
    "feedback": "feedback", 
    "end": END
})

workflow.add_conditional_edges("review", should_continue, {
    "execute": "execute", 
    "go_to_coding": "coding", 
    "end": END
})

workflow.add_conditional_edges("execute", should_continue, {
    "go_to_feedback": "feedback", 
    "end": END
})

workflow.add_conditional_edges("feedback", should_continue, {
    "end": END,
    "go_to_coding": "coding",
    "reassess": "retrieval" 
})

app = workflow.compile()

if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = False

def create_graph():
    G = nx.DiGraph()
    nodes = ["retrieval", "coding", "review", "execute", "feedback", "END"]
    edges = [
        ("retrieval", "coding", "go_to_coding"),
        ("retrieval", "execute", "go_to_execute"),
        ("retrieval", "feedback", "go_to_feedback"),
        ("retrieval", "END", "complete"),
        ("retrieval", "END", "end"),
        ("retrieval", "retrieval", "reassess"),
        ("coding", "review", "review"),
        ("coding", "execute", "execute"),
        ("coding", "feedback", "feedback"),
        ("coding", "END", "end"),
        ("review", "execute", "execute"),
        ("review", "coding", "go_to_coding"),
        ("review", "END", "end"),
        ("execute", "feedback", "go_to_feedback"),
        ("execute", "END", "end"),
        ("feedback", "END", "end"),
        ("feedback", "coding", "go_to_coding"),
        ("feedback", "retrieval", "reassess")
    ]
    G.add_nodes_from(nodes)
    for start, end, label in edges:
        G.add_edge(start, end, label=label)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, font_weight='bold', arrows=True)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    return plt

button_style = """
<style>
.big-btn {
    display: inline-block;
    margin: 1px;
    padding: 0.5em 3em;
    border: 0.16em solid rgba(255,255,255,0);
    border-radius: 2em;
    box-sizing: border-box;
    text-decoration: none;
    font-family: 'Roboto',sans-serif;
    font-weight: 300;
    color: #FFFFFF;
    text-align: center;
    transition: all 0.2s;
}
.big-btn-positive {
    background-color: #4CAF50;
}
.big-btn-negative {
    background-color: #F44336;
}
.big-btn:hover {
    border-color: #FFFFFF;
}
</style>
"""
st.markdown(f"<h1 style='text-align: center; margin-bottom: 1rem;'>Langgraph Project App</h1>", unsafe_allow_html=True)

st.markdown(button_style, unsafe_allow_html=True)

input_query = st.text_input("Input Query", "")

if st.button('Execute'):
    inputs = {
        "messages": [HumanMessage(content=input_query)],
        "code": "",
        "data": input_query,  
        "execution_result": True,
        "user_feedback": "positive"
    }

    try:
        output = app.invoke(inputs)
    except:
        pass

    DATA_AI_OUTPUT_QUERY = vector_store.get_data_key()
    DATA_USER_INPUT_QUERY = input_query + "\n , write a python code that manupulate the data and show the  chart if needed also print the output in table ,use hapi fhir server baseURL is http://hapi.fhir.org/baseR4/ and return 100 object" 

    print(DATA_AI_OUTPUT_QUERY)

    st.markdown("<h2>Input Message</h2>", unsafe_allow_html=True)
    st.write(DATA_USER_INPUT_QUERY, input_query)
    print("Input Query:", input_query)
    
    st.markdown("<h2>AI Prompt Output</h2>", unsafe_allow_html=True)
    st.write(DATA_AI_OUTPUT_QUERY)
    
    st.markdown("<h2>Graph Nodes</h2>", unsafe_allow_html=True)
    fig, ax = plt.subplots()
    create_graph()
    st.pyplot(fig)

    if not st.session_state.feedback_given:
        st.markdown("<h2>User Feedback:</h2>", unsafe_allow_html=True)
        # feedback = st.selectbox("Choose your feedback:", ["Select", "Positive", "Negative"])
        
        # if feedback == "Positive":
        #     db_data = load_database()
        #     st.json(db_data) 
        # elif feedback == "Negative":
        #     json_file_path = 'db.json'
        #     key_to_remove = str(vector_store.curr_key)
        #     if key_to_remove in data:
        #         del data[key_to_remove]
        #     st.write("Because of Negative Feedback, the Prompt hasn't been recorded.")

        feedback = st.selectbox("Choose your feedback:", ["", "Positive", "Negative"], key='feedback_selector') 
        if feedback == "Positive":
            with open('db.json') as file:
                db_data = json.load(file)
            st.json(db_data)  
            st.session_state.feedback_given = True
        elif feedback == "Negative":
            json_file_path = 'db.json'
            key_to_remove = str(vector_store.curr_key)
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            if key_to_remove in data:
                del data[key_to_remove]
            with open(json_file_path, 'w') as file:
                json.dump(data, file, indent=4)
            st.write("Because of Negative Feedback, the Prompt hasn't been recorded.")
            st.session_state.feedback_given = True
