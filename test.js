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
import { ChatPromptTemplate } from "@langchain/core/prompts";
import * as z from "zod";

async function main() {
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

  console.log("🔍 Loading documents...");

  const docs = await Promise.all(
    urls.map((url) => new CheerioWebBaseLoader(url).load())
  );
  const docsFlat = docs.flat();

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });
  const splitDocs = await textSplitter.splitDocuments(docsFlat);

  console.log(`Split into ${splitDocs.length} chunks of documents.`);

  // -------------------------------------------

  // 2- Create a retriever tool
  /**
   * 1- use in-memory vector store , openai embeddings
   * 2- Create a retriever tool using LangChain’s prebuilt createRetrieverTool
   */
  console.log("Creating vector store and retriever...");

  const vector = await MemoryVectorStore.fromDocuments(
    splitDocs,

    new OpenAIEmbeddings({
      configuration: {
        baseURL: "http://127.0.0.1:1234/v1",
      },
      model: "text-embedding-nomic-embed-text-v1.5",
    })
  );
  const retriever = vector.asRetriever();

  const tool = createRetrieverTool(retriever, {
    name: "retrieve_blog_posts",
    description:
      "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
  });
  const tools = [tool];

  // -------------------------------------------

  // 3- Generate query (build node - edges)
  /**
   * the generateQueryOrRespond method will decide whether to use the retriever tool or directly respond
   */
  async function generateQueryOrRespond(state) {
    const { messages } = state;

    // console.log(
    //   "\n Current conversation state:",
    //   messages.map(
    //     (m) => `${m.constructor.name}: ${m.content.substring(0, 50)}...`
    //   )
    // );

    const model = new ChatOpenAI({
      configuration: {
        baseURL: "http://127.0.0.1:1234/v1",
      },
      model: "granite-4.0-h-tiny",
      temperature: 0,
      maxTokens: 500,
    }).bindTools(tools);

    // Initial model call to decide tool usage
    let response = await model.invoke(messages);

    // console.log("DEBUG: Initial model response:", {
    //   content: response.content?.substring(0, 100) + "...",
    //   tool_calls: response.tool_calls?.map((tc) => ({
    //     name: tc.name,
    //     args: tc.args,
    //   })),
    // });

    // Process tool calls if any
    if (response.tool_calls?.length > 0) {
      // console.log("DEBUG: Tool calls detected! Processing...");

      // Execute all tools in parallel
      const toolResponses = await Promise.all(
        response.tool_calls.map(async (toolCall) => {
          //

          // console.log(
          //   `DEBUG: Executing tool: ${toolCall.name} with args:`,
          //   toolCall.args
          // );

          const toolObj = tools.find((t) => t.name === toolCall.name);
          let toolResult = "Tool execution error";

          try {
            if (toolObj) {
              toolResult = await toolObj.invoke(toolCall.args);

              // console.log(
              //   `DEBUG: Tool result (${toolResult.length} chars):`,

              //   typeof toolResult === "string"
              //     ? toolResult.substring(0, 100) + "..."
              //     : JSON.stringify(toolResult).substring(0, 100) + "..."
              // );
            }
          } catch (err) {
            // console.error(`Error in ${toolCall.name}:`, err.message);
            toolResult = `Error: ${err.message}`;
          }

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

      // console.log("DEBUG: Sending tool responses to model for final answer...");
      response = await model.invoke(updatedMessages);

      // console.log(
      //   "DEBUG: Final response:",
      //   response.content?.substring(0, 200) + "..."
      // );
    } else {
      console.log("DEBUG: No tool calls made, direct response");
    }

    return { messages: [response] };
  }
  console.log("🤖 Running test query...");

  // const input = {
  //   messages: [
  //     new HumanMessage(
  //       "What are the different types of LLM agents according to Lilian Weng?"
  //     ),
  //   ],
  // };
  // const result = await generateQueryOrRespond(input);
  // console.log("Final result:", result.messages[0]);

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

  const input = {
    messages: [
      new HumanMessage(
        "What does Lilian Weng say about types of reward hacking?"
      ),
      new AIMessage({
        tool_calls: [
          {
            type: "tool_call",
            name: "retrieve_blog_posts",
            args: { query: "types of reward hacking" },
            id: "1",
          },
        ],
      }),
      new ToolMessage({
        content: "meow",
        tool_call_id: "1",
      }),
    ],
  };
  const result = await gradeDocuments(input);
  console.log("Grading result (should be 'rewrite'):", result);

  //

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
    return {
      messages: [response],
    };
  }

  // 2- Run the rewrite node

  const input2 = {
    messages: [
      new HumanMessage(
        "What does Lilian Weng say about types of reward hacking?"
      ),
      new AIMessage({
        content: "",
        tool_calls: [
          {
            id: "1",
            name: "retrieve_blog_posts",
            args: { query: "types of reward hacking" },
            type: "tool_call",
          },
        ],
      }),
      new ToolMessage({ content: "meow", tool_call_id: "1" }),
    ],
  };

  const response = await rewrite(input2);
  console.log(response.messages[0].content);
}

main().catch(console.error);
