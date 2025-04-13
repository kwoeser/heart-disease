# PersonalWebsite

## Overview

This repository contains the code for my personal website.

**Description:** None

**Primary Language:** JavaScript

**Topics:** (Add relevant topics here, such as "portfolio", "react", "javascript", "personal-website")

This website is built with React and utilizes various libraries for different functionalities. It's designed to showcase my projects, skills, and experience.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/<your-username>/PersonalWebsite.git
    cd PersonalWebsite
    ```

2.  **Install dependencies:**

    ```bash
    npm install  # Or yarn install if you prefer yarn
    ```

## Usage

1.  **Start the development server:**

    ```bash
    npm run dev  # Or yarn dev if you use yarn
    ```

    This will start the Vite development server, typically at `http://localhost:5173/`.  Open this address in your browser to view the website.

2.  **Build for production:**

    ```bash
    npm run build # Or yarn build
    ```

    This will create an optimized production build in the `dist` directory.  You can then deploy this directory to a web server (e.g., Netlify, Vercel, AWS S3).

3.  **Preview the production build locally:**

    ```bash
    npm run preview # Or yarn preview
    ```

    This will serve the production build locally, allowing you to test it before deployment.

## Dependencies

The following dependencies are used in this project:

*   **`@emailjs/browser`**:  Used for sending emails from the frontend using EmailJS.
*   **`axios`**:  Used for making HTTP requests.
*   **`cheerio`**:  A fast, flexible, and lean implementation of core jQuery designed specifically for the server.  Likely used for web scraping or parsing HTML.
*   **`cors`**:  Used for enabling Cross-Origin Resource Sharing (CORS) on the server-side.
*   **`csv-parser`**:  Used for parsing CSV files.
*   **`express`**:  A minimal and flexible Node.js web application framework.
*   **`framer-motion`**:  A production-ready motion library for React.
*   **`lucide-react`**: A collection of beautiful, customizable icons.
*   **`node-fetch`**:  A lightweight module that brings the `fetch` API to Node.js.
*   **`react`**:  A JavaScript library for building user interfaces.
*   **`react-dom`**:  Provides DOM-specific methods that are useful for managing the app at the top level.
*   **`react-router-dom`**:  Provides the necessary components for implementing routing in a React application.
*   **`rss-parser`**:  A parser for RSS feeds.
*   **`esbuild`**: An extremely fast JavaScript bundler.


## Development Dependencies

The following devDependencies are used for development purposes:

*   **`@eslint/js`**: Configurable ESLint rule set.
*   **`@types/react`**:  TypeScript definitions for React.
*   **`@types/react-dom`**:  TypeScript definitions for React DOM.
*   **`@vitejs/plugin-react`**:  A Vite plugin that provides React support.
*   **`eslint`**:  A JavaScript linting tool.
*   **`eslint-plugin-react`**:  ESLint plugin for React specific linting rules.
*   **`eslint-plugin-react-hooks`**:  ESLint plugin for React Hooks.
*   **`eslint-plugin-react-refresh`**: ESLint plugin to ensure Fast Refresh works correctly.
*   **`globals`**:  Provides an environment to create javascript files.
*   **`vite`**:  A fast build tool for modern web applications.
*   **`esbuild`**: An extremely fast JavaScript bundler.

## Contributing

Contributions are welcome! If you find a bug or have an idea for a new feature, please open an issue or submit a pull request.

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with descriptive messages.
4.  Submit a pull request to the `main` branch.

Please ensure that your code adheres to the project's coding style and that all tests pass before submitting a pull request.  Run `npm run lint` before submitting.

## License

[Specify the license here, e.g., MIT License]

Copyright (c) [Year] [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
