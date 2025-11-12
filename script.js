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
import {
  Annotation,
  StateGraph,
  START,
  END,
  messagesStateReducer,
} from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";

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
    configuration: {
      baseURL: "http://127.0.0.1:1234/v1",
    },
    model: "granite-4.0-h-tiny",
    temperature: 0,
    maxTokens: 500,
  }).bindTools(tools);

  const cleanMessages = messages.filter(
    (m) =>
      m instanceof HumanMessage ||
      m instanceof AIMessage ||
      m instanceof ToolMessage
  );

  let response = await model.invoke(cleanMessages);

  console.log("AIMessage tool_calls:", response.tool_calls?.length);

  // Process tool calls

  if (response.tool_calls?.length > 0) {
    // Execute all tools in parallel

    const toolResponses = await Promise.all(
      response.tool_calls.map(async (toolCall) => {
        //

        const toolObj = tools.find((t) => t.name === toolCall.name);
        let toolResult = "Tool execution error";

        try {
          if (toolObj) {
            toolResult = await toolObj.invoke(toolCall.args);
          }
        } catch (err) {
          toolResult = `Error: ${err.message}`;
        }

        console.log("AIMessage tool_calls:", response.tool_calls?.length);

        return new ToolMessage({
          content:
            typeof toolResult === "string"
              ? toolResult
              : JSON.stringify(toolResult),
          tool_call_id: toolCall.id,
        });
      })
    );

    const updatedMessages = [...messages, response, ...toolResponses];

    response = await model.invoke(updatedMessages);
  } else {
    console.log("DEBUG: No tool calls made, direct response");
  }

  return { messages: [...messages, response] };
}

console.log("🤖 Running test query...");

// -------------------------------------------
// -------------------------------------------

// 4- Grading Docs
/***
 * 1- Add a node — gradeDocuments — to determine whether the retrieved
 *    documents are relevant to the question. We will use a model with
 *    structured output using Zod for document grading. We’ll also add a
 *    conditional edge — checkRelevance — that checks the grading result
 *    and returns the name of the node to go to (generate or rewrite):
 *
 * 2- Run this with irrelevant documents in the tool response:
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

const gradeDocumentsSchema = z.object({
  binaryScore: z.string().describe("Relevance score 'yes' or 'no'"),
});

async function gradeDocuments(state) {
  const { messages } = state;

  const model = new ChatOpenAI({
    configuration: { baseURL: "http://127.0.0.1:1234/v1" },
    model: "granite-4.0-h-tiny",
    temperature: 0,
  }).withStructuredOutput(gradeDocumentsSchema);

  // 1
  const score = await prompt.pipe(model).invoke({
    question: messages.at(0).content,
    context: messages.at(-1).content,
  });

  if (score.binaryScore === "yes") return "generate";
  return "rewrite";
}

// 2

// const gradeInput = {
//   messages: [
//     new HumanMessage(
//       "What does Lilian Weng say about types of reward hacking?"
//     ),
//     new AIMessage({
//       tool_calls: [
//         {
//           type: "tool_call",
//           name: "retrieve_blog_posts",
//           args: { query: "types of reward hacking" },
//           id: "1",
//         },
//       ],
//     }),
//     new ToolMessage({
//       content: "meow",
//       tool_call_id: "1",
//     }),
//   ],
// };
// const gradeResult = await gradeDocuments(gradeInput);
// console.log("Grading result (should be 'rewrite'):", gradeResult);

// -------------------------------------------
// -------------------------------------------

// 5- Rewrite question
/**
 * 1- Build the rewrite node. The retriever tool can return potentially
 *    irrelevant documents, which indicates a need to improve the original
 *    user question. To do so, we will call the rewrite node:
 */

const rewritePrompt = ChatPromptTemplate.fromTemplate(
  `Look at the input and try to reason about the underlying semantic intent / meaning. \n
  Here is the initial question:
  \n ------- \n
  {question}
  \n ------- \n
  Formulate an improved question:`
);

async function rewrite(state) {
  const { messages } = state;
  const question = messages.at(0)?.content;

  const model = new ChatOpenAI({
    configuration: { baseURL: "http://127.0.0.1:1234/v1" },
    model: "granite-4.0-h-tiny",
    temperature: 0,
  });

  const response = await rewritePrompt.pipe(model).invoke({ question });
  return { messages: [...messages, response] };
}

// 2- Run the rewrite node

// const rewriteInput = {
//   messages: [
//     new HumanMessage(
//       "What does Lilian Weng say about types of reward hacking?"
//     ),
//     new AIMessage({
//       content: "",
//       tool_calls: [
//         {
//           id: "1",
//           name: "retrieve_blog_posts",
//           args: { query: "types of reward hacking" },
//           type: "tool_call",
//         },
//       ],
//     }),
//     new ToolMessage({ content: "meow", tool_call_id: "1" }),
//   ],
// };

// const rewriteResponse = await rewrite(rewriteInput);
// console.log(rewriteResponse.messages[0].content);

// -------------------------------------------
// -------------------------------------------

// 6- Generating answer
/***
 * Build generate node: if we pass the grader checks, we can generate
 * the final answer based on the original question and the retrieved context:
 */

async function generate(state) {
  const { messages } = state;
  const question = messages.at(0)?.content;
  const context = messages.at(-1)?.content;

  const prompt = ChatPromptTemplate.fromTemplate(
    `You are an assistant for question-answering tasks.
      Use the following pieces of retrieved context to answer the question.
      If you don't know the answer, just say that you don't know.
      Use three sentences maximum and keep the answer concise.
      Question: {question}
      Context: {context}`
  );

  const model = new ChatOpenAI({
    configuration: { baseURL: "http://127.0.0.1:1234/v1" },
    model: "granite-4.0-h-tiny",
    temperature: 0,
  });

  const ragChain = prompt.pipe(model);

  const response = await ragChain.invoke({
    context,
    question,
  });

  return { messages: [...messages, response] };
}

// const generateInput = {
//   messages: [
//     new HumanMessage(
//       "What does Lilian Weng say about types of reward hacking?"
//     ),
//     new AIMessage({
//       content: "",
//       tool_calls: [
//         {
//           id: "1",
//           name: "retrieve_blog_posts",
//           args: { query: "types of reward hacking" },
//           type: "tool_call",
//         },
//       ],
//     }),
//     new ToolMessage({
//       content:
//         "reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering",
//       tool_call_id: "1",
//     }),
//   ],
// };

// const generateResponse = await generate(generateInput);
// console.log(generateResponse.messages[0].content);

// -------------------------------------------
// -------------------------------------------

// 7- Assemble the graph
/**
 * Now we’ll assemble all the nodes and edges into a complete graph:
 *  * Start with a generateQueryOrRespond and determine if we need to call the retriever tool
 *  * Route to next step using a conditional edge:
 *    - If generateQueryOrRespond returned tool_calls, call the retriever tool to retrieve context
 *    - Otherwise, respond directly to the user
 *  * Grade retrieved document content for relevance to the question (gradeDocuments) and route to next step:
 *    - If not relevant, rewrite the question using rewrite and then call generateQueryOrRespond again
 *    - If relevant, proceed to generate and generate final response using the @[ToolMessage] with the retrieved document context
 */

// Create a ToolNode for the retriever
const toolNode = new ToolNode(tools);

// Helper function to determine if we should retrieve
function shouldRetrieve(state) {
  const { messages } = state;
  const lastMessage = messages.at(-1);

  if (AIMessage.isInstance(lastMessage) && lastMessage.tool_calls.length) {
    return "retrieve";
  }
  return END;
}

const GraphState = Annotation.Root({
  messages: Annotation({
    reducer: messagesStateReducer,
    default: () => [],
  }),
  grade: Annotation.String,
});

// Define the graph
const builder = new StateGraph(GraphState)
  .addNode("generateQueryOrRespond", generateQueryOrRespond)
  .addNode("retrieve", toolNode)
  .addNode("gradeDocuments", gradeDocuments)
  .addNode("rewrite", rewrite)
  .addNode("generate", generate)
  // Add edges
  .addEdge(START, "generateQueryOrRespond")
  // Decide whether to retrieve
  .addConditionalEdges("generateQueryOrRespond", shouldRetrieve)
  .addEdge("retrieve", "gradeDocuments")
  // Edges taken after grading documents
  .addConditionalEdges(
    "gradeDocuments",
    // Route based on grading decision
    (state) => {
      // The gradeDocuments function returns either "generate" or "rewrite"
      const lastMessage = state.messages.at(-1);
      return lastMessage.content === "generate" ? "generate" : "rewrite";
    }
  )
  .addEdge("generate", END)
  .addEdge("rewrite", "generateQueryOrRespond");

// Compile
const graph = builder.compile();

// 8- Run the agentic RAG
const inputs = {
  messages: [
    new HumanMessage(
      "reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering"
    ),
  ],
};

for await (const output of await graph.stream(inputs)) {
  for (const [key, value] of Object.entries(output)) {
    const lastMsg = output[key].messages[output[key].messages.length - 1];
    console.log(`Output from node: '${key}'`);
    console.log({
      type: lastMsg._getType(),
      content: lastMsg.content,
      tool_calls: lastMsg.tool_calls,
    });
    console.log("---\n");
  }
}
