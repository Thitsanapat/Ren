
const express = require("express");
const fs = require("fs");
const use = require("@tensorflow-models/universal-sentence-encoder");
const tf = require("@tensorflow/tfjs-node");

const app = express();
const port = 3000;

// Middleware to parse JSON bodies
app.use(express.json());

// Load the Universal Sentence Encoder model once, for performance reasons
let model;
(async () => {
  model = await use.load();
  console.log("Universal Sentence Encoder model loaded.");
})();

// Function to calculate cosine similarity
function cosineSimilarity(vecA, vecB) {
  const dotProduct = vecA.reduce((sum, a, idx) => sum + a * vecB[idx], 0);
  const normA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const normB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (normA * normB);
}

// Endpoint to handle clustering requests
app.post("/cluster-questions", async (req, res) => {
  try {
    const { questionsWithIds } = req.body;
    if (!questionsWithIds) {
      return res.status(400).json({ error: "questionsWithIds is required." });
    }

    const questions = Object.values(questionsWithIds);
    const ids = Object.keys(questionsWithIds);

    // Step 1: Convert questions to embeddings
    const embeddings = await model.embed(questions);

    // Step 2: Calculate similarity matrix
    const similarityMatrix = [];
    const embeddingsArray = await embeddings.array();
    for (let i = 0; i < embeddingsArray.length; i++) {
      similarityMatrix[i] = [];
      for (let j = 0; j < embeddingsArray.length; j++) {
        similarityMatrix[i][j] = cosineSimilarity(embeddingsArray[i], embeddingsArray[j]);
      }
    }

    // Step 3: Cluster using similarity threshold
    const threshold = 0.9; // Adjust as needed
    const thresholdClusters = {};
    const visited = new Set();

    for (let i = 0; i < questions.length; i++) {
      if (visited.has(i)) continue;
      thresholdClusters[i] = [{ id: ids[i], question: questions[i] }];
      visited.add(i);
      for (let j = i + 1; j < questions.length; j++) {
        if (!visited.has(j) && similarityMatrix[i][j] > threshold) {
          thresholdClusters[i].push({ id: ids[j], question: questions[j] });
          visited.add(j);
        }
      }
    }

    // Step 4: Return the clusters
    return res.json({ clusters: thresholdClusters });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: "An error occurred while processing the request." });
  }
});

// Start the server
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});

