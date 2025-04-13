# Personal Website

A personal website built with React, designed to showcase my skills, projects, and experiences.

## Overview

This repository contains the source code for my personal website. It's a single-page application (SPA) built with React and utilizes various libraries for styling, animations, and data fetching.  The website includes sections for an introduction, projects, a blog (fetched from RSS feed), and contact information.

## Built With

![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoWidth=60) ![Node.js](https://img.shields.io/badge/Node.js-339933?style=for-the-badge&logo=node.js&logoWidth=60) ![Vite](https://img.shields.io/badge/Vite-B1B1B1?style=for-the-badge&logo=vite&logoColor=FF4949&logoWidth=60) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoWidth=60) ![ESLint](https://img.shields.io/badge/ESLint-4B32C3?style=for-the-badge&logo=eslint&logoWidth=60) ![Framer Motion](https://img.shields.io/badge/Framer_Motion-0055FF?style=for-the-badge&logo=framer&logoColor=white&logoWidth=60)

## Installation

To run this project locally, you'll need to have Node.js and npm (or yarn) installed.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/<your_github_username>/PersonalWebsite.git
    cd PersonalWebsite
    ```

2.  **Install dependencies:**

    ```bash
    npm install
    # or
    yarn install
    ```

## Usage

1.  **Start the development server:**

    ```bash
    npm run dev
    # or
    yarn dev
    ```

    This will start the Vite development server, and you can view the website in your browser at the address provided in the console (usually `http://localhost:5173`).

2.  **Build for production:**

    ```bash
    npm run build
    # or
    yarn build
    ```

    This will create a `dist` directory containing the production-ready files.

3.  **Preview the production build:**

     ```bash
     npm run preview
     # or
     yarn preview
     ```

## Dependencies

```json
{
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
}
```

## Contributing

Contributions are welcome!  If you find a bug or have a suggestion for improvement, please open an issue or submit a pull request.

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with descriptive messages.
4.  Push your branch to your fork.
5.  Submit a pull request to the main repository.

## License

This project is licensed under the [MIT License](LICENSE) - see the `LICENSE` file for details.
