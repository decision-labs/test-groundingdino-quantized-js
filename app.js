// Define the HTTP server
import http from "http";
import querystring from "querystring";
import url from "url";
import { MyClassificationPipeline, SegmentAnythingSingleton } from "./pipeline.js";

const server = http.createServer();
const hostname = "127.0.0.1";
const port = 3000;

// Listen for requests made to the server
server.on("request", async (req, res) => {
  // Parse the request URL
  const parsedUrl = url.parse(req.url);

  // Extract the query parameters
  const { model_name, text, image_uri } = querystring.parse(parsedUrl.query);
  if (!model_name) {
    res.statusCode = 400;
    res.end(JSON.stringify({ error: "model_name is required must be sam or classifier" }));
    return;
  }

  let classifier, sam_model, sam_processor;
  if (model_name === "sam") {
    [sam_model, sam_processor] = await SegmentAnythingSingleton.getInstance();
  }
  if (model_name === "classifier") {
    classifier = await MyClassificationPipeline.getInstance();
  }
  console.log(text, model_name, image_uri);


  // Set the response headers
  res.setHeader("Content-Type", "application/json");

  let response;
  if (parsedUrl.pathname === "/classify" && text) {
    // const classifier = await MyClassificationPipeline.getInstance();
    response = await classifier(text);
    res.statusCode = 200;
  } else {
    response = { error: "Bad request" };
    res.statusCode = 400;
  }

  // Send the JSON response
  res.end(JSON.stringify(response));
});

server.listen(port, hostname, () => {
  console.log(`Server running at http://${hostname}:${port}/`);
});
