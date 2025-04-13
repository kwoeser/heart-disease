# Personal Website

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the source code for my personal website. The site is designed to showcase my skills, projects, and experience. It's built using modern web technologies and is intended to be responsive and accessible.

## Features

*   **About Me:** A brief introduction and overview of my background.
*   **Projects:** A showcase of my past and current projects, with details and links where applicable.
*   **Skills:** A list of my technical skills and areas of expertise.
*   **Blog/Articles:** (If applicable) A section for sharing my thoughts and insights on various topics.
*   **Contact:** A contact form or links to my social media profiles for easy communication.

## Built With

This project is built using the following technologies:

*   **React:** [![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)  A JavaScript library for building user interfaces.
*   **Node.js:** [![NodeJS](https://img.shields.io/badge/node.js-6DA55F?style=for-the-badge&logo=node.js&logoColor=white)](https://nodejs.org/en/) A JavaScript runtime built on Chrome's V8 JavaScript engine.
*   **Vite:** [![Vite](https://img.shields.io/badge/vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev/)  Next Generation Frontend Tooling.
*   **Express:** [![Expressjs](https://img.shields.io/badge/express.js-%23404d59.svg?style=for-the-badge&logo=express&logoColor=%2361DAFB)](https://expressjs.com/)  A fast, unopinionated, minimalist web framework for Node.js.
*   **Framer Motion:** [![Framer Motion](https://img.shields.io/badge/FramerMotion-0055FF?style=for-the-badge&logo=framer&logoColor=white)](https://www.framer.com/motion/) A production-ready motion library for React.
*   **Lucide React:** [![Lucide](https://img.shields.io/badge/Lucide-1A202C?style=for-the-badge&logo=lucide&logoColor=white)](https://lucide.dev/)  A library of beautiful, consistent icons.
*   **Axios:** A Promise based HTTP client for the browser and node.js.
*   **EmailJS:**  A service for sending emails from your client-side applications.
*   **Cheerio:** A fast, flexible, and lean implementation of core jQuery designed specifically for the server.
*   **CORS:** A node.js package for providing a Connect/Express middleware that can be used to enable CORS with various options.
*   **CSV-Parser:** A streaming CSV parser for Node.js
*   **RSS-Parser:** A Node.js RSS parser.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/PersonalWebsite.git
    cd PersonalWebsite
    ```

2.  **Install dependencies:**

    ```bash
    npm install
    # or
    yarn install
    # or
    pnpm install
    ```

## Usage

1.  **Start the development server:**

    ```bash
    npm run dev
    # or
    yarn dev
    # or
    pnpm dev
    ```

    This will start the development server, and you can access the website in your browser at the address provided (usually `http://localhost:5173/`).

2.  **Build for production:**

    ```bash
    npm run build
    # or
    yarn build
    # or
    pnpm build
    ```

    This will create a production-ready build of the website in the `dist` directory.  You can then deploy the contents of this folder to your hosting provider.

## Dependencies

The project's dependencies, as defined in `package.json`, are as follows:

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
    "rss-parser": "^3.13.0"
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
    "vite": "^6.2.5"
  }
}
```

## Contributing

Contributions are welcome! If you find a bug or have a suggestion, please open an issue or submit a pull request.

Here's how to contribute:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with descriptive commit messages.
4.  Push your changes to your fork.
5.  Submit a pull request to the main repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
