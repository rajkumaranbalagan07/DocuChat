import { PineconeClient } from "@pinecone-database/pinecone";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import * as dotenv from "dotenv";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { OpenAI } from "langchain/llms/openai";
import * as readline from 'readline';

dotenv.config();

// Step 3
async function vectorStore() {
    try {
        const client = new PineconeClient();
        await client.init({
            apiKey: process.env.PINECONE_API_KEY ? process.env.PINECONE_API_KEY : "",
            environment: process.env.PINECONE_ENVIRONMENT ? process.env.PINECONE_ENVIRONMENT : ""
        });
        const index = client.Index(process.env.PINECONE_INDEX ? process.env.PINECONE_INDEX : "");

        const vectorStore = await PineconeStore.fromExistingIndex(new OpenAIEmbeddings(),
            {
                pineconeIndex: index, textKey: 'text', 
                // namespace: process.env.PINECONE_NAME_SPACE ?
                    // process.env.PINECONE_NAME_SPACE : ""
            }
        );
        return vectorStore;
    } catch (error) {
        console.error("Error While trying to connect with the PineCone" + error)
    }
}


const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

{context}

Question: {question}
Helpful answer in markdown:`;

// Step 4
async function query(query: string) {
    try {
        // console.log("query is=====",query);
        const pineCoreStore = await vectorStore();
        if (pineCoreStore) {
            const model = new OpenAI({
                openAIApiKey: process.env.OPENAI_API_KEY,
                temperature: 0,
                modelName: 'gpt-3.5-turbo'
            })
            const chain = ConversationalRetrievalQAChain
                .fromLLM(model, pineCoreStore?.asRetriever(), {
                    qaTemplate: QA_PROMPT,
                    questionGeneratorTemplate: CONDENSE_PROMPT,
                    returnSourceDocuments: true
                });

            const answer = await chain.call({
                question: query,
                chat_history: [],
            });
            // console.log(answer?.text);            
            return answer;
        }
    } catch (error) {
        console.error("Error While trying to connect with the PineCone" + error)
    }
}



const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

async function askQuestion() {
    rl.question('Please enter your input: ', async (input: string) => {
        // Process the input here
        // console.log(`You entered: ${input}`);

        const answer = await query(input);

        console.log(`=============================Answer==============================================================================`);
        console.log(`Your Response: ${answer?.text}`);
        console.log(`=================================================================================================================`);

        // Check if the input is 'exit', and close the readline interface if it is
        if (input.toLowerCase() === 'exit') {
            rl.close();
        } else {
            // Otherwise, ask the question again
            await askQuestion();
        }
    });
}

await askQuestion();

rl.on('close', () => {
    console.log('Exiting...');
    process.exit(0);
});