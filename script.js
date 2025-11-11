import dotenv from "dotenv";
dotenv.config();
import { createRetrieverTool } from "@langchain/classic/agents/toolkits";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import {
  HumanMessage,
  SystemMessage,
  AIMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatOpenAI } from "@langchain/openai";
import * as z from "zod";
import { ChatPromptTemplate } from "@langchain/core/prompts";

// -------------------------------------------
// -------------------------------------------

// 1- Preprocess documents
/**
 * 1- fetch documents to use in our RAG system
 * 2- Split the fetched documents into smaller chunks for indexing into our vectorstore:
 */
const urls = [
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
  "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
];

// 1
const docs = await Promise.all(
  urls.map((url) => new CheerioWebBaseLoader(url).load())
);

// 2
const docsFlat = docs.flat();
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 50,
});
const splitDocs = await textSplitter.splitDocuments(docsFlat);

// -------------------------------------------
// -------------------------------------------

// 2- Create a retriever tool
/**
 * 1- use in-memory vector store , openai embeddings
 * 2- Create a retriever tool using LangChain’s prebuilt createRetrieverTool
 */
console.log("Creating vector store , retriever...");

const vector = await MemoryVectorStore.fromDocuments(
  splitDocs,
  new OpenAIEmbeddings({
    // openAIApiKey: "not-needed",
    configuration: {
      baseURL: "http://127.0.0.1:1234/v1", // LM Studio
    },
    model: "text-embedding-nomic-embed-text-v1.5",
  })
);

//
const retriever = vector.asRetriever();
const tool = createRetrieverTool(retriever, {
  name: "retrieve_blog_posts",
  description:
    "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
});
const tools = [tool];

// -------------------------------------------
// -------------------------------------------

// 3- Generate query (build node - edges)
/**
 * the generateQueryOrRespond method will decide whether to use the retriever tool or directly respond
 */

async function generateQueryOrRespond(state) {
  const { messages } = state;

  const model = new ChatOpenAI({
    configuration: { baseURL: "http://127.0.0.1:1234/v1" },
    model: "granite-4.0-h-tiny",
    temperature: 0,
  }).bindTools(tools);

  let response = await model.invoke(messages);

  // Process tool calls
  if (response.tool_calls && response.tool_calls.length > 0) {
    for (const toolCall of response.tool_calls) {
      //
      const toolObj = tools.find((t) => t.name === toolCall.name);
      let toolResult = "Tool not found";

      try {
        if (toolObj) {
          toolResult = await toolObj.invoke(toolCall.args);
        }
      } catch (err) {
        toolResult = `Error running tool ${toolCall.name}: ${err.message}`;
      }

      const toolMessage = new ToolMessage({
        content:
          typeof toolResult === "string"
            ? toolResult
            : JSON.stringify(toolResult),
        tool_call_id: toolCall.id,
      });

      response = await model.invoke([...messages, response, toolMessage]);
    }
  }

  return { messages: [response] };
}

console.log("🤖 Running test query...");

// -------------------------------------------
// -------------------------------------------

// 4- Grading Docs
/***
 * Add a node — gradeDocuments — to determine whether the retrieved
 * documents are relevant to the question. We will use a model with
 * structured output using Zod for document grading. We’ll also add a
 * conditional edge — checkRelevance — that checks the grading result
 * and returns the name of the node to go to (generate or rewrite):
 */

const prompt = ChatPromptTemplate.fromTemplate(
  `You are a grader assessing relevance of retrieved docs to a user question.
  Here are the retrieved docs:
  \n ------- \n
  {context}
  \n ------- \n
  Here is the user question: {question}
  If the content of the docs are relevant to the users question, score them as relevant.
  Give a binary score 'yes' or 'no' score to indicate whether the docs are relevant to the question.
  Yes: The docs are relevant to the question.
  No: The docs are not relevant to the question.`
);
