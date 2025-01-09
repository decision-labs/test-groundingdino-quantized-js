// Define the HTTP server
import cors from "cors";
import express from "express";
import {
  MyClassificationPipeline,
  SegmentAnythingSingleton,
  GroundingDinoSingleton,
} from "./pipeline.js";

const app = express();
const hostname = "0.0.0.0";
const port = 3001;

// Enable CORS for all routes
app.use(cors());
app.use(express.json());

app.get("/ping", (req, res) => {
  res.json({ status: "ok" });
});

app.get("/", async (req, res) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);

  // Extract the query parameters
  let { model_name, text, image_uri } = req.query;

  if (!model_name) {
    return res.status(400).json({
      error: "model_name is required must be sam or classifier",
    });
    return;
  }

  let classifier, sam_model, sam_processor;
  if (model_name === "sam") {
    [sam_model, sam_processor] = await SegmentAnythingSingleton.getInstance();
  }
  if (model_name === "classifier") {
    classifier = await MyClassificationPipeline.getInstance();
  }
  if (model_name === "object-detection") {
    if (!Array.isArray(text)) {
      if (!text.endsWith(".")) text = text + ".";
      text = [text];
    }
    const grounding_dino = await GroundingDinoSingleton.getInstance();
    const features = await grounding_dino(image_uri, text, { threshold: 0.3 });
    return res.json(features);
  }

  console.log(text, model_name, image_uri);

  let response;

  if (req.path === "/classify" && text) {
    response = await classifier(text);
    return res.json(response);
  } else {
    return res.status(400).json({ error: "Bad request" });
  }
});

app.listen(port, hostname, () => {
  console.log(`Server running at http://${hostname}:${port}/`);
});
