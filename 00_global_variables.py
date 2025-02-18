# Databricks notebook source
# Packages required by all code.
# Versions of Databricks code are not locked since Databricks ensures changes are backwards compatible.
# Versions of open source packages are locked since package authors often make backwards compatible changes
%pip install -qqqq -U \
  databricks-vectorsearch databricks-agents pydantic databricks-sdk mlflow mlflow-skinny `# For agent & data pipeline code` \
  pypdf==4.1.0  `# PDF parsing` \
  markdownify==0.12.1  `# HTML parsing` \
  pypandoc_binary==1.13  `# DOCX parsing` \
  transformers==4.41.1 torch==2.3.0 tiktoken==0.7.0 langchain-text-splitters==0.2.0. `# get_recursive_character_text_splitter`

# Restart to load the packages into the Python environment
dbutils.library.restartPython()

# COMMAND ----------

user_email = spark.sql("SELECT current_user() as username").collect()[0].username
user_name = user_email.split("@")[0].replace(".", "").lower()[:35]
print(f"user_name {user_name}")
AGENT_NAME = "vip_agent_app"
UC_MODEL_NAME = f"det_dev.vip_ops_schema.{AGENT_NAME}"

EVALUATION_SET_FQN = f"det_dev.vip_ops_schema.{AGENT_NAME}_evaluation_set_final"
#"det_dev.vip_ops_schema.vip_knowledge_center"
# MLflow experiment name
# Using the same MLflow experiment for a single app allows you to compare runs across Notebooks
MLFLOW_EXPERIMENT_NAME = f"/Users/{user_email}/{AGENT_NAME}"

# Data pipeline MLflow run name
POC_DATA_PIPELINE_RUN_NAME = "data_pipeline_poc_final"
# Chain MLflow run name
POC_CHAIN_RUN_NAME = "agent_poc_final"

# COMMAND ----------

# Vector Search endpoint where index is loaded
# If this does not exist, it will be created
VECTOR_SEARCH_ENDPOINT = f"vip_knowledge_center_final"

# Source location for documents
# You need to create this location and add files
SOURCE_UC_VOLUME = f"/Volumes/det_dev/vip_ops_schema/vip_knowledge_center"

# Names of the output Delta Tables tables & Vector Search index

# Delta Table with the parsed documents and extracted metadata
DOCS_DELTA_TABLE = f"det_dev.vip_ops_schema.vip_knowledge_center"

# Chunked documents that are loaded into the Vector Index
CHUNKED_DOCS_DELTA_TABLE = "det_dev.vip_ops_schema.vip_knowledge_center_chuncked_final"

# Vector Index
VECTOR_INDEX_NAME = "det_dev.vip_ops_schema.vip_knowledge_center_docs_index_final"



# Embedding model endpoint. The list of off-the-shelf embeddings can be found here:
# https://docs.databricks.com/en/machine-learning/foundation-models/index.html
EMBEDDING_MODEL_ENDPOINT = "databricks-bge-large-en"
# EMBEDDING_MODEL_ENDPOINT = "ep-embeddings-small"
# EMBEDDING_MODEL_ENDPOINT = "bge-test"

# COMMAND ----------

print("--user info--")
print(f"user_name {user_name}")

print("--agent--")
print(f"AGENT_NAME {AGENT_NAME}")
# print(f"UC_CATALOG {UC_CATALOG}")
# print(f"UC_SCHEMA {UC_SCHEMA}")
print(f"UC_MODEL_NAME {UC_MODEL_NAME}")

print()
print("--evaluation config--")
print(f"EVALUATION_SET_FQN {EVALUATION_SET_FQN}")
print(f"MLFLOW_EXPERIMENT_NAME {MLFLOW_EXPERIMENT_NAME}")
print(f"POC_DATA_PIPELINE_RUN_NAME {POC_DATA_PIPELINE_RUN_NAME}")
print(f"POC_CHAIN_RUN_NAME {POC_CHAIN_RUN_NAME}")