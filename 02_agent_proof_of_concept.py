# Databricks notebook source
# MAGIC %md
# MAGIC # Agent proof of concept.
# MAGIC
# MAGIC By the end of this notebook, you will have created a POC of your Agent that you can interact with, and ask questions.
# MAGIC
# MAGIC This means:
# MAGIC - We will have a mlflow model registered in the "Models" tab on the Databricks menu on the left. Models that are registered are just assets that can be instantiated from another notebook, but are not served on an endpoint. These models can be invoked with `mlflow.invoke()`.
# MAGIC - We will have a served model registered in the "Serving" tab on the Databricks menu on the left. This means that the model is served and can be accessed via a UI or a REST API for anyone in the workspace.
# MAGIC

# COMMAND ----------

# Versions of Databricks code are not locked since Databricks ensures changes are backwards compatible.
%pip install -qqqq -U databricks-agents databricks-vectorsearch databricks-sdk mlflow mlflow-skinny
# Restart to load the packages into the Python environment
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import the global configuration

# COMMAND ----------

# MAGIC %run ./00_global_variables

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent configuration

# COMMAND ----------

# MAGIC %run ./agents/agent_config

# COMMAND ----------

retriever_config = RetrieverToolConfig(
    vector_search_index=VECTOR_INDEX_NAME,
    # VECTOR_INDEX_NAME = "det_dev.vip_ops_schema.vip_knowledge_center_docs_index"
    # vector_search_index="det_dev.vip_ops_schema.vip_knowledge_center_docs_index",
    vector_search_schema=RetrieverSchemaConfig(
        primary_key="chunk_id",
        chunk_text="content_chunked",
        document_uri="doc_uri",
        additional_metadata_columns=[],
    ),
    parameters=RetrieverParametersConfig(num_results=16, query_type="ann"),
    vector_search_threshold=0.1,
    chunk_template="Passage text: {chunk_text}\nPassage metadata: {metadata}\n\n",
    prompt_template="""Use the following pieces of retrieved context to answer the question.\nOnly use the passages from context that are relevant to the query to answer the question, ignore the irrelevant passages \n\nContext: {context}""",
    # Prompt template citing sources
    # prompt_template="""Use the following pieces of retrieved context to answer the question.\nOnly use the passages from context that are relevant to the query to answer the question, ignore the irrelevant passages.  When responding, cite your source, referring to the passage by the columns in the passage's metadata.\n\nContext: {context}""",
    tool_description_prompt="Search for documents that are relevant to a user's query about the Daily Operations Steps: Detailed, step-by-step guides for daily tasks involved in the job. These steps are critical for ensuring consistency and accuracy in operations. Preoperative Plans: Segmentation: Detailed instructions on how to segment medical images or data, specifying anatomical or procedural segments. Landmarking: Guidance on identifying and marking key landmarks, which could be anatomical or surgical points of interest. Implants: Specifics on the selection, placement, and considerations for surgical implants. This might include material types, sizes, and compatibility. Targeter Settings: Detailed settings for surgical navigation or targeting systems used in procedures. Documentation: Instructions on how to document each step of the preoperative and operative processes for compliance, quality control, and educational purposes. Special Procedures: Metal: Information regarding the use and handling of metal in procedures, including types of metal used, corrosion resistance, and biocompatibility. Jira: Integration with or usage of Jira for project management, tracking issues, or managing changes in operational procedures. Portal Procedures: Accessing, contributing to, or retrieving data from specialized portals or systems that might be internal or industry-specific.",
)

# `llm_endpoint_name`: Model Serving endpoint with the LLM for your Agent.
#     - Either an [Foundational Models](https://docs.databricks.com/en/machine-learning/foundation-models/index.html) Provisioned Throughput / Pay-Per-Token or [External Model](https://docs.databricks.com/en/generative-ai/external-models/index.html) of type `/llm/v1/chat` with support for [function calling](https://docs.databricks.com/en/machine-learning/model-serving/function-calling.html).  Supported models: `databricks-meta-llama-3-70b-instruct` or any of the Azure OpenAI / OpenAI models.

llm_config = LLMConfig(
    # https://docs.databricks.com/en/machine-learning/foundation-models/index.html
    llm_endpoint_name="databricks-meta-llama-3-1-70b-instruct",
    # Define a template for the LLM prompt.  This is how the RAG chain combines the user's question and the retrieved context.
    llm_system_prompt_template=(
        """You are a helpful assistant that answers questions by calling tools.  Provide responses ONLY based on the information from tools that are explictly specified to you.  If you do not have a relevant tool for a question, respond with 'Sorry, I'm not trained to answer that question'."""
    ),
    # Parameters that control how the LLM responds.
    llm_parameters=LLMParametersConfig(temperature=0.01, max_tokens=1500),
)

agent_config = AgentConfig(
    retriever_tool=retriever_config,
    llm_config=llm_config,
    input_example={
        "messages": [
            {
                "role": "user",
                "content": "What is RAG?",
            },
        ]
    },
)

agent_config.dict()

# COMMAND ----------

# MAGIC %md
# MAGIC **Test the query results of a question with the endpoint det_dev.vip_ops_schema.vip_knowledge_center_docs_index2**

# COMMAND ----------

import os
from databricks.vector_search.client import VectorSearchClient

workspace_url = os.environ.get("WORKSPACE_URL")
sp_client_id = os.environ.get("SP_CLIENT_ID")
sp_client_secret = os.environ.get("SP_CLIENT_SECRET")

vsc = VectorSearchClient(
    workspace_url=workspace_url,
    service_principal_client_id=sp_client_id,
    service_principal_client_secret=sp_client_secret
)

index = vsc.get_index(endpoint_name="vip_knowledge_center", index_name="det_dev.vip_ops_schema.vip_knowledge_center_docs_index2")

index.similarity_search(num_results=3, columns=["content_chunked"], query_text="Should I use an augment if bss is less than 85%?")

# COMMAND ----------

# MAGIC %run ./validators/validate_agent_config

# COMMAND ----------

validate_retriever_config(retriever_config)
validate_llm_config(llm_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save the agent config

# COMMAND ----------

save_agent_config(agent_config.dict(), './agents/generated_configs/agent.yaml')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set the MLflow experiement name
# MAGIC
# MAGIC Used to store the Agent's model

# COMMAND ----------

import mlflow

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import the Agent's code

# COMMAND ----------

# MAGIC %run ./agents/function_calling_agent_w_retriever_tool

# COMMAND ----------

# MAGIC %md
# MAGIC ## Iteratively vibe check & adjust the Agent's configuration
# MAGIC
# MAGIC If you need to adjust the Agent's code, you can do so in the `./agents/function_calling_agent_w_retriever_tool` Notebook

# COMMAND ----------

# Load the Agent
agent = AgentWithRetriever()

# COMMAND ----------



# Example for testing multiple turns of converastion

# 1st turn of converastion
first_turn_input = {
    "messages": [
        {"role": "user", "content": f"Should I use an augment if bss is less than 85%?"},
    ]
}

response = agent.predict(model_input=first_turn_input)
print("Response 1st turn")
print(response["content"])

# 2nd turn of converastion
new_messages = response["messages"]
new_messages.append({"role": "user", "content": f"Should I use an augment if bss is less than 85%?"})
# print(type(new_messages))
second_turn_input = {"messages": new_messages}
response = agent.predict(model_input=second_turn_input)
print("Response 2nd turn")
print(response["content"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the Agent POC to the Review App

# COMMAND ----------

from mlflow.models.resources import (
    DatabricksVectorSearchIndex,
    DatabricksServingEndpoint,
)
from mlflow.models.signature import ModelSignature
from mlflow.models.rag_signatures import StringResponse, ChatCompletionRequest
import yaml
from databricks import agents
from databricks import vector_search

databricks_resources = [
    DatabricksServingEndpoint(endpoint_name=llm_config.llm_endpoint_name),
    DatabricksVectorSearchIndex(index_name=retriever_config.vector_search_index),
]

# Specify the full path to the Agent notebook
model_file = "agents/function_calling_agent_w_retriever_tool"
model_path = os.path.join(os.getcwd(), model_file)

with mlflow.start_run(run_name=POC_CHAIN_RUN_NAME):
    model_info = mlflow.pyfunc.log_model(
        python_model=model_path,
        model_config=agent_config.dict(),
        artifact_path="agent",
        input_example=agent_config.input_example,
        resources=databricks_resources,
        signature=ModelSignature(
            inputs=ChatCompletionRequest(),
            outputs=StringResponse(),
        ),
    )

# COMMAND ----------

### Invoke the logged model
model = mlflow.pyfunc.load_model(model_info.model_uri)
model.predict(agent_config.input_example)

# COMMAND ----------

# Use Unity Catalog as the model registry
mlflow.set_registry_uri("databricks-uc")

# Register the model to the Unity Catalog
uc_registered_model_info = mlflow.register_model(
    model_uri=model_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

from databricks import agents
import time
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate


# Deploy to enable the Review APP and create an API endpoint
deployment_info = agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version)

# Wait for the Review App to be ready
print("\nWaiting for endpoint to deploy.  This can take 15 - 20 minutes.", end="")

print(f"\n\nReview App: {deployment_info.review_app_url}")