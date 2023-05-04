import * as dotenv from "dotenv";
import { PineconeClient } from "@pinecone-database/pinecone";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";

dotenv.config();

async function loadPDF() {
  try {
    const loader = new PDFLoader("src/books/science.pdf");
    const docs = await loader.load();
    return docs;
  } catch (error) {
    console.log("Error loading PDF:", error);
    return [];
  }
}

async function splitDocuments(documents:any) {
  try {
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200
    });
    const docs = await splitter.splitDocuments(documents);
    return docs;
  } catch (error) {
    console.log("Error splitting documents:", error);
    return [];
  }
}

async function createVectorStore() {
  try {
    const client = new PineconeClient();
    await client.init({
      apiKey: process.env.PINECONE_API_KEY || "",
      environment: process.env.PINECONE_ENVIRONMENT || ""
    });

    const index = client.Index(process.env.PINECONE_INDEX || "");

    const documents = await loadPDF();
    const splitDocs = await splitDocuments(documents || []);
    await PineconeStore.fromDocuments(splitDocs, new OpenAIEmbeddings(), {
      pineconeIndex: index,
      textKey: 'text'
    });
    console.log("Done");
  } catch (error) {
    console.error("Error creating vector store:", error);
  }
}

// (async () => {
//   await createVectorStore();
// })();

await createVectorStore();
