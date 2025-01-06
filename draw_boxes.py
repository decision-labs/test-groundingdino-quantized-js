import requests
import cv2
import numpy as np

def fetch_model_output(url, params):
    try:
        # Make the GET request
        response = requests.get(url, params=params)

        # Check if the response was successful
        if response.status_code == 200:
            # Parse and return the JSON response
            return response.json()
        else:
            print(f"Request failed with status code {response.status_code}: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the GET request: {e}")
        return None

# URL of the model endpoint
url = "http://localhost:3000/"
params = {
    "model_name": "object-detection",
    "text": "tree",
    "image_uri": "https://content.satimagingcorp.com/static/galleryimages/Satellite-Image-Paris-Pont-des-Arts-bridge.jpg"
}

# Fetch the model output
model_output = fetch_model_output(url, params)

if model_output:
    print("Model output fetched successfully!")

    # Load the image from the provided URI
    image_uri = params["image_uri"]
    response = requests.get(image_uri)
    image_array = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Draw bounding boxes on the image
    for detection in model_output:
        box = detection["box"]
        score = detection.get("score", 0)
        label = detection.get("label", "Object")

        # Get box coordinates
        xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]

        # Draw rectangle
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Add label and score
        text = f"{label} {score:.2f}"
        cv2.putText(image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save or display the output image
    output_path = "output_with_boxes.jpg"
    cv2.imwrite(output_path, image)

    # Optional: Display the image
    # cv2.imshow("Detected Objects", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print(f"Output saved to {output_path}")
else:
    print("Failed to retrieve model output.")
