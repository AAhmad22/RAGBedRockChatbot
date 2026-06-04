"""CDK stack for the RAGBedRockChatbot.

Provisions, using the AWS Generative AI CDK Constructs (awslabs):
  - an S3 bucket for source documents
  - a Bedrock Knowledge Base backed by an auto-provisioned OpenSearch
    Serverless vector collection + index (the construct also wires the
    encryption / network / data-access policies and the KB service role)
  - an S3 data source for the Knowledge Base
  - a Lambda (FastAPI via Mangum) behind API Gateway to serve /chat

Writing and synthesising this is free. ``cdk deploy`` creates real resources
and starts billing - notably the OpenSearch Serverless collection, which bills
hourly even when idle, so tear the stack down when you are done.
"""

from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_apigateway as apigw,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_s3 as s3,
)
from cdklabs.generative_ai_cdk_constructs import bedrock
from constructs import Construct


class RagStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs: object) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # --- Source documents -------------------------------------------
        docs_bucket = s3.Bucket(
            self,
            "DocsBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            removal_policy=RemovalPolicy.DESTROY,  # easy teardown for a demo
            auto_delete_objects=True,
        )

        # --- Knowledge Base + OpenSearch Serverless ---------------------
        # VectorKnowledgeBase auto-provisions the OpenSearch Serverless
        # collection, the vector index, the access/encryption/network
        # policies, and the KB service role - the parts that are very
        # error-prone to hand-wire with raw L1 constructs.
        knowledge_base = bedrock.VectorKnowledgeBase(
            self,
            "KnowledgeBase",
            embeddings_model=bedrock.BedrockFoundationModel.TITAN_EMBED_TEXT_V2_1024,
            instruction=(
                "Answer using only the indexed documents. If the documents do "
                "not contain the answer, say you do not have enough information."
            ),
        )

        data_source = bedrock.S3DataSource(
            self,
            "DataSource",
            bucket=docs_bucket,
            knowledge_base=knowledge_base,
            data_source_name="documents",
            chunking_strategy=bedrock.ChunkingStrategy.FIXED_SIZE,
        )

        # --- Serving layer: FastAPI on Lambda behind API Gateway --------
        # NOTE: this asset ships only the app source. Before the function
        # will actually run, its dependencies (fastapi, mangum, pydantic)
        # must be bundled in - via a Lambda layer or a container image. The
        # KB data layer above deploys and works without this; for a quick
        # demo you can run the FastAPI app locally against the deployed KB.
        handler = lambda_.Function(
            self,
            "ChatHandler",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="main.handler",
            code=lambda_.Code.from_asset("../app"),
            timeout=Duration.seconds(30),
            memory_size=512,
            environment={"KNOWLEDGE_BASE_ID": knowledge_base.knowledge_base_id},
        )
        handler.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:Retrieve",
                    "bedrock:RetrieveAndGenerate",
                    "bedrock:InvokeModel",
                ],
                resources=["*"],  # TODO: scope to the KB and model ARNs
            )
        )

        api = apigw.LambdaRestApi(self, "ChatApi", handler=handler, proxy=True)

        # --- Outputs (consumed by ingestion + the frontend) ------------
        CfnOutput(self, "DocsBucketName", value=docs_bucket.bucket_name)
        CfnOutput(self, "KnowledgeBaseId", value=knowledge_base.knowledge_base_id)
        CfnOutput(self, "DataSourceId", value=data_source.data_source_id)
        CfnOutput(self, "ChatApiUrl", value=api.url)
