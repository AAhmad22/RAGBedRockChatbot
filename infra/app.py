#!/usr/bin/env python3
import aws_cdk as cdk
from stacks.rag_stack import RagStack

app = cdk.App()
RagStack(app, "RagBedrockChatbotStack")
app.synth()
