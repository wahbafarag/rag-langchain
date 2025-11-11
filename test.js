import dotenv from "dotenv";
dotenv.config();
import { createRetrieverTool } from "@langchain/classic/agents/toolkits";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatOpenAI } from "@langchain/openai";
import { ToolMessage } from "@langchain/core/messages";

async function main() {
  // -------------------------------------------
  // 1- Preprocess documents
  const urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
  ];

  // fetch documents to use in our RAG system
  const docs = await Promise.all(
    urls.map((url) => new CheerioWebBaseLoader(url).load())
  );

  // Split the fetched documents into smaller chunks for indexing into our vectorstore:
  const docsFlat = docs.flat();
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });
  const splitDocs = await textSplitter.splitDocuments(docsFlat);
  console.log(`Split into ${splitDocs.length} chunks of documents.`);

  // -------------------------------------------

  // 2- Create a retriever tool

  // use in-memory vector store , openai embeddings
  console.log("Creating vector store , retriever...");

  const vector = await MemoryVectorStore.fromDocuments(
    splitDocs,
    new OpenAIEmbeddings({
      openAIApiKey: "not-needed",
      configuration: {
        baseURL: "http://127.0.0.1:1234/v1",
      },
      model: "text-embedding-nomic-embed-text-v1.5",
    })
  );

  const retriever = vector.asRetriever();

  // Create a retriever tool using LangChain’s prebuilt createRetrieverTool
  const tool = createRetrieverTool(retriever, {
    name: "retrieve_blog_posts",
    description:
      "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
  });
  const tools = [tool];

  // -------------------------------------------

  // 3- Generate query (build node - edges)
  // the generateQueryOrRespond method will decide whether to use the retriever tool or directly respond

  async function generateQueryOrRespond(state) {
    const { messages } = state;

    const model = new ChatOpenAI({
      configuration: { baseURL: "http://127.0.0.1:1234/v1" },
      model: "granite-4.0-h-tiny",
      temperature: 0,
    }).bindTools(tools);

    let response = await model.invoke(messages);

    console.log("First model response:", {
      content: response.content,
      hasToolCalls: response.tool_calls?.length > 0,
      toolCalls: response.tool_calls,
    });

    // Process tool calls
    if (response.tool_calls && response.tool_calls.length > 0) {
      for (const toolCall of response.tool_calls) {
        console.log("Executing tool call:", toolCall.name, toolCall.args);

        const toolObj = tools.find((t) => t.name === toolCall.name);
        let toolResult = "Tool not found";

        try {
          if (toolObj) {
            toolResult = await toolObj.invoke(toolCall.args);
          }
        } catch (err) {
          toolResult = `Error running tool ${toolCall.name}: ${err.message}`;
        }

        console.log(
          `Tool result length: ${String(toolResult).length} characters`
        );

        const toolMessage = new ToolMessage({
          content:
            typeof toolResult === "string"
              ? toolResult
              : JSON.stringify(toolResult),
          tool_call_id: toolCall.id,
        });

        response = await model.invoke([...messages, response, toolMessage]);

        console.log("Final response (after tool):", {
          content: response.content,
          hasToolCalls: response.tool_calls?.length > 0,
        });
      }
    }

    return { messages: [response] };
  }

  // Test the function
  // console.log("🤖 Running test query...");

  // const input = { messages: [new HumanMessage("hello!")] };
  // const result = await generateQueryOrRespond(input);
  // console.log(result.messages[0]);

  // const input = {
  //   messages: [
  //     new SystemMessage(
  //       "You are a research assistant who can use tools to answer questions accurately using Lilian Weng’s blog."
  //     ),

  //     new HumanMessage(
  //       "What does Lilian Weng say about types of reward hacking?"
  //     ),
  //   ],
  // };
  // const result = await generateQueryOrRespond(input);
  // console.log(result.messages[0]);

  //
}
main().catch(console.error);
