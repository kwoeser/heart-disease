# PersonalWebsite

A personal website built with React and various other technologies. This project showcases my skills and experience.

## Overview

This repository contains the source code for my personal website. It's built using a modern JavaScript stack and includes features like a portfolio section, a blog (utilizing RSS feeds), and a contact form. The website is designed to be responsive and accessible across different devices. It also features web scraping and CSV parsing for different content.

## Built With

*   **React:** [![React Badge](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)
*   **Vite:** [![Vite Badge](https://img.shields.io/badge/vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev/)
*   **Node.js:** [![NodeJS Badge](https://img.shields.io/badge/node.js-6DA55F?style=for-the-badge&logo=node.js&logoColor=white)](https://nodejs.org/)
*   **Express:** [![ExpressJS Badge](https://img.shields.io/badge/express.js-%23404d59.svg?style=for-the-badge&logo=express&logoColor=%2361DAFB)](https://expressjs.com/)
*   **Framer Motion:** [![Framer Motion Badge](https://img.shields.io/badge/Framer_Motion-white?style=for-the-badge&logo=framer&logoColor=blue)](https://www.framer.com/motion/)
*   **Lucide React:** [![Lucide React Badge](https://img.shields.io/badge/Lucide_React-000000?style=for-the-badge&logo=lucide&logoColor=white)](https://lucide.dev/)
*   **Axios:** [![Axios Badge](https://img.shields.io/badge/axios-000000?style=for-the-badge&logo=axios&logoColor=white)](https://axios-http.com/)

## Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    ```

2.  Navigate to the project directory:

    ```bash
    cd PersonalWebsite
    ```

3.  Install the dependencies:

    ```bash
    npm install
    ```

## Usage

1.  Start the development server:

    ```bash
    npm run dev
    ```

    This will start the application in development mode. Open your browser and navigate to the address provided by Vite (usually `http://localhost:5173`).

2.  Build the application for production:

    ```bash
    npm run build
    ```

    This will create a `dist` directory containing the production-ready build of the website.

3.  Preview the production build locally:

    ```bash
    npm run preview
    ```

## Dependencies

The project utilizes the following dependencies (as specified in `package.json`):

```json
{
  "name": "my-portfolio",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "lint": "eslint .",
    "preview": "vite preview"
  },
  "dependencies": {
    "@emailjs/browser": "^4.4.1",
    "axios": "^1.8.1",
    "cheerio": "^1.0.0",
    "cors": "^2.8.5",
    "csv-parser": "^3.2.0",
    "express": "^4.21.2",
    "framer-motion": "^12.4.7",
    "lucide-react": "^0.475.0",
    "node-fetch": "^3.3.2",
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "react-router-dom": "^7.2.0",
    "rss-parser": "^3.13.0",
    "esbuild": ">=0.25.0"
  },
  
  "devDependencies": {
    "@eslint/js": "^9.19.0",
    "@types/react": "^19.0.8",
    "@types/react-dom": "^19.0.3",
    "@vitejs/plugin-react": "^4.3.4",
    "eslint": "^9.19.0",
    "eslint-plugin-react": "^7.37.4",
    "eslint-plugin-react-hooks": "^5.0.0",
    "eslint-plugin-react-refresh": "^0.4.18",
    "globals": "^15.14.0",
    "vite": "^6.2.5",
    "esbuild": ">=0.25.0"
  }
}
```

*   **@emailjs/browser:** Used for sending emails directly from the browser.
*   **axios:** HTTP client for making API requests.
*   **cheerio:** Fast, flexible, and lean implementation of core jQuery designed specifically for the server.
*   **cors:** Middleware for enabling Cross-Origin Resource Sharing (CORS) with various options.
*   **csv-parser:** Parses CSV data.
*   **express:** Fast, unopinionated, minimalist web framework for Node.js.
*   **framer-motion:** Motion library for React.
*   **lucide-react:** Beautifully simple, pixel-perfect icons for React.
*   **node-fetch:** A light-weight module that brings `window.fetch` to Node.js.
*   **react:** JavaScript library for building user interfaces.
*   **react-dom:** Entry point to the DOM and server rendering APIs for React.
*   **react-router-dom:** DOM bindings for React Router.
*   **rss-parser:** RSS and Atom feed parser.

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes.
4.  Commit your changes with descriptive commit messages.
5.  Push your branch to your forked repository.
6.  Create a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
