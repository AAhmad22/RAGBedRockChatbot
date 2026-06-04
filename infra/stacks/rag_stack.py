"""CDK stack for the RAGBedRockChatbot.

This is a *stub* that lays out the resources and their relationships. The
sections marked TODO are the parts that need real configuration before
`cdk deploy` will produce a working system (notably: the OpenSearch
Serverless vector index mapping, and the Bedrock embedding/foundation
model ARNs for your region).

NOTE: Bedrock Knowledge Bases involve several fiddly L1 (Cfn*) constructs.
If you'd rather not wire them by hand, the community library
`@cdklabs/generative-ai-cdk-constructs` provides higher-level constructs
for Bedrock Knowledge Bases + OpenSearch Serverless. This stub uses L1
constructs to keep dependencies minimal and the moving parts visible.
"""

from __future__ import annotations

from aws_cdk import (
    RemovalPolicy,
    Stack,
)
from aws_cdk import (
    aws_apigateway as apigw,
)
from aws_cdk import (
    aws_bedrock as bedrock,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_lambda as lambda_,
)
from aws_cdk import (
    aws_opensearchserverless as aoss,
)
from aws_cdk import (
    aws_s3 as s3,
)
from constructs import Construct


class RagStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs: object) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # --- 1. Source document bucket -------------------------------------
        docs_bucket = s3.Bucket(
            self,
            "DocsBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            removal_policy=RemovalPolicy.DESTROY,  # portfolio project: easy teardown
            auto_delete_objects=True,
        )

        # --- 2. OpenSearch Serverless vector collection --------------------
        # TODO: add the encryption / network / data-access policies that
        # OpenSearch Serverless requires, and create the vector index with
        # the correct dimension for your embedding model (e.g. 1024 for
        # Titan Text Embeddings v2).
        vector_collection = aoss.CfnCollection(
            self,
            "VectorCollection",
            name="ragbedrock-vectors",
            type="VECTORSEARCH",
        )

        # --- 3. IAM role assumed by the Bedrock Knowledge Base -------------
        kb_role = iam.Role(
            self,
            "KnowledgeBaseRole",
            assumed_by=iam.ServicePrincipal("bedrock.amazonaws.com"),
        )
        docs_bucket.grant_read(kb_role)
        # TODO: grant aoss:APIAccessAll on the collection + bedrock invoke on
        # the embedding model to kb_role.

        # --- 4. Bedrock Knowledge Base + S3 data source --------------------
        # TODO: set EMBEDDING_MODEL_ARN for your region, and fill in the
        # OpenSearch Serverless field mapping (vector / text / metadata).
        embedding_model_arn = (
            f"arn:aws:bedrock:{self.region}::foundation-model/"
            "amazon.titan-embed-text-v2:0"
        )
        knowledge_base = bedrock.CfnKnowledgeBase(
            self,
            "KnowledgeBase",
            name="ragbedrock-kb",
            role_arn=kb_role.role_arn,
            knowledge_base_configuration=bedrock.CfnKnowledgeBase.KnowledgeBaseConfigurationProperty(
                type="VECTOR",
                vector_knowledge_base_configuration=bedrock.CfnKnowledgeBase.VectorKnowledgeBaseConfigurationProperty(
                    embedding_model_arn=embedding_model_arn,
                ),
            ),
            storage_configuration=bedrock.CfnKnowledgeBase.StorageConfigurationProperty(
                type="OPENSEARCH_SERVERLESS",
                opensearch_serverless_configuration=bedrock.CfnKnowledgeBase.OpenSearchServerlessConfigurationProperty(
                    collection_arn=vector_collection.attr_arn,
                    vector_index_name="ragbedrock-index",
                    field_mapping=bedrock.CfnKnowledgeBase.OpenSearchServerlessFieldMappingProperty(
                        vector_field="vector",  # TODO: match your index
                        text_field="text",
                        metadata_field="metadata",
                    ),
                ),
            ),
        )

        bedrock.CfnDataSource(
            self,
            "S3DataSource",
            name="ragbedrock-s3-source",
            knowledge_base_id=knowledge_base.attr_knowledge_base_id,
            data_source_configuration=bedrock.CfnDataSource.DataSourceConfigurationProperty(
                type="S3",
                s3_configuration=bedrock.CfnDataSource.S3DataSourceConfigurationProperty(
                    bucket_arn=docs_bucket.bucket_arn,
                ),
            ),
        )

        # --- 5. Lambda handler (FastAPI app) -------------------------------
        # TODO: package app/ with its dependencies (e.g. via a Docker image
        # or a Lambda layer) before this will deploy a working function.
        handler = lambda_.Function(
            self,
            "ChatHandler",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="main.handler",
            code=lambda_.Code.from_asset("../app"),  # TODO: bundle deps
            environment={
                "KNOWLEDGE_BASE_ID": knowledge_base.attr_knowledge_base_id,
                # TODO: GENERATION_MODEL_ID, etc.
            },
        )
        handler.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:RetrieveAndGenerate", "bedrock:Retrieve"],
                resources=["*"],  # TODO: scope to the KB ARN
            )
        )

        # --- 6. API Gateway in front of the Lambda -------------------------
        apigw.LambdaRestApi(
            self,
            "ChatApi",
            handler=handler,
            proxy=True,
        )
