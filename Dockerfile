# Use an official Node.js runtime as a parent image
FROM node:20-bullseye

# Set the working directory in the container
WORKDIR /usr/src/app

# Install git
# RUN apk add --no-cache git

# Clone the transformers.js repository and build it
RUN git clone https://github.com/huggingface/transformers.js.git && \
    cd transformers.js && \
    git checkout add-grounding-dino && \
    npm install && \
    npm run build && \
    cd ..

# Copy package.json and yarn.lock to the working directory
COPY package.json yarn.lock ./

# Install dependencies
RUN yarn install

# Copy the rest of the application code to the working directory
COPY . .

# Expose the port the app runs on
EXPOSE 3001

# Command to run the application
CMD ["npx", "nodemon", "app.js"]