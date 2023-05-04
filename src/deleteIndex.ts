import * as dotenv from "dotenv";
import { PineconeClient } from "@pinecone-database/pinecone";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";

dotenv.config();



async function createVectorStore() {
  try {
    const client = new PineconeClient();
    await client.init({
      apiKey: process.env.PINECONE_API_KEY || "",
      environment: process.env.PINECONE_ENVIRONMENT || ""
    });
    const index = client.Index(process.env.PINECONE_INDEX || "");
    const deleted = await index.delete1();
    console.log({deleted});    
    console.log("Deleted",);
  } catch (error) {
    console.error("Error creating vector store:", error);
  }
}

(async () => {
  await createVectorStore();
})();
