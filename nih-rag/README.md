# NIH Powered LLM Using RAG

This is the code to a chatbot that uses the NIH ODS Fact Sheets (https://ods.od.nih.gov/api/) to generate grounded answers.

The project uses LangChain with a technique called Retrieval Augmented Generation (RAG) to provide a Large Language Model (LLM) with useful context to base results on.

### Walktrhough

Video Walkthrough Link: 

pip install command:

`pip install --upgrade --quiet  langchain langchain-community langchainhub langchain-openai langchain-chroma bs4`

### Sources Mentioned

- [Quickstart for Question Answering](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/)
- [Use OpenAI Embeddings with Chroma](https://python.langchain.com/v0.2/docs/integrations/vectorstores/chroma/#use-openai-embeddings)
- [How to Use QA Sources](https://python.langchain.com/v0.2/docs/how_to/qa_sources/)
- [Chroma Telemetry Documentation](https://docs.trychroma.com/telemetry)
