import axios from "axios";
import sharp from "sharp";
import { SamModel, AutoProcessor, RawImage } from "@xenova/transformers";

async function fetchModelOutput(url, params) {
  try {
    const response = await axios.get(url, { params });
    if (response.status === 200) {
      return response.data;
    } else {
      console.error(
        `Request failed with status code ${response.status}: ${response.data}`
      );
      return null;
    }
  } catch (error) {
    console.error("An error occurred during the GET request:", error);
    return null;
  }
}

const doSegment = async (boxes, image_uri) => {
  try {
    const image = await RawImage.read(image_uri);
    // Load SAM model and processor
    const model = await SamModel.from_pretrained("Xenova/sam-vit-base");
    const processor = await AutoProcessor.from_pretrained(
      "Xenova/sam-vit-base"
    );

    const masks = [];
    // Process each bounding box
    for (let i = 0; i < boxes.length; i++) {
      const [xmin, ymin, xmax, ymax] = boxes[i];

      const centerX = Math.floor((xmin + xmax) / 2);
      const centerY = Math.floor((ymin + ymax) / 2);
      const input_points = [[[centerX, centerY]]];

      const inputs = await processor(image, input_points);
      const outputs = await model(inputs);

      const processedMasks = await processor.post_process_masks(
        outputs.pred_masks,
        inputs.original_sizes,
        inputs.reshaped_input_sizes
      );

      // Store mask data for canvas rendering
      masks.push({
        mask: processedMasks[0][0],
        box: { xmin, ymin, xmax, ymax },
      });

      //save the mask to a file
      const masked_image = RawImage.fromTensor(processedMasks[0][0].mul(255));
      masked_image.save(`mask_${i}.png`);
    }

    return masks;
  } catch (error) {
    console.error("Error in segmentation:", error);
  }
};

async function drawBoxes() {
  // URL of the model endpoint
  const url = "http://localhost:3001/";
  const params = {
    model_name: "object-detection",
    text: "tree",
    image_uri:
      "https://content.satimagingcorp.com/static/galleryimages/Satellite-Image-Paris-Pont-des-Arts-bridge.jpg",
  };

  // Fetch the model output
  const modelOutput = await fetchModelOutput(url, params);

  const boxex = modelOutput.map((detection) => {
    return Object.values(detection.box);
  });

  await doSegment(boxex, params.image_uri);

  if (modelOutput) {
    console.log("Model output fetched successfully!");

    try {
      // Load the image from the provided URI
      const response = await axios.get(params.image_uri, {
        responseType: "arraybuffer",
      });
      const image = sharp(response.data);

      // Get image metadata
      const metadata = await image.metadata();
      const { width = 0, height = 0 } = metadata;

      // Create SVG with bounding boxes
      const svgString = `
                <svg width="${width}" height="${height}">
                    ${modelOutput
                      .map((detection) => {
                        const { box, score = 0, label = "Object" } = detection;
                        const { xmin, ymin, xmax, ymax } = box;
                        return `
                            <rect x="${xmin}" y="${ymin}" 
                                  width="${xmax - xmin}" height="${ymax - ymin}"
                                  fill="none" stroke="lime" stroke-width="2"/>
                            <text x="${xmin}" y="${ymin - 5}" 
                                  font-family="Arial" font-size="16" fill="lime">
                                ${label} ${score.toFixed(2)}
                            </text>
                        `;
                      })
                      .join("")}
                </svg>`;

      // Composite the SVG onto the image
      await image
        .composite([
          {
            input: Buffer.from(svgString),
            top: 0,
            left: 0,
          },
        ])
        .toFile("output_with_boxes1.jpg");

      console.log("Output saved to output_with_boxes.jpg");
    } catch (error) {
      console.error("Error processing image:", error);
    }
  } else {
    console.log("Failed to retrieve model output.");
  }
}

// Run the script
drawBoxes().catch(console.error);
