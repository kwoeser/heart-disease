# PersonalWebsite

## Overview

This repository contains the code for my personal website. It's a platform to showcase my projects, skills, and experience. The site is built with modern web technologies, providing a responsive and engaging user experience.

## Features

*   **Portfolio Showcase:** Highlights my projects with descriptions, images, and links.
*   **Skills Section:** Lists my technical skills and areas of expertise.
*   **Blog (Optional):** Contains articles and posts on relevant topics. (Currently inactive).
*   **Contact Form:** Allows visitors to easily reach out to me.
*   **Responsive Design:** Adapts to different screen sizes and devices.

## Built With

![React](https://img.shields.io/badge/React-20232A?style=flat-square&logo=react) ![Vite](https://img.shields.io/badge/Vite-B73BFE?style=flat-square&logo=vite) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript) ![Node.js](https://img.shields.io/badge/Node.js-339933?style=flat-square&logo=node.js) ![Express.js](https://img.shields.io/badge/Express.js-000000?style=flat-square&logo=express)
## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/YOUR_USERNAME/PersonalWebsite.git
    ```

2.  Navigate to the project directory:

    ```bash
    cd PersonalWebsite
    ```

3.  Install dependencies:

    ```bash
    npm install
    ```

## Usage

1.  Start the development server:

    ```bash
    npm run dev
    ```

    This will start the application in development mode, typically at `http://localhost:5173/`.

2.  Build for production:

    ```bash
    npm run build
    ```

    This will create a production-ready build in the `dist` directory.

3.  To run the express backend (if applicable):

    ```bash
    node server.js
    ```
    or
    ```bash
    nodemon server.js
    ```

## Configuration

The following environment variables are used (if backend is enabled):

*   `PORT`: The port on which the server listens (defaults to 3000).
*   `EMAILJS_SERVICE_ID`: The EmailJS service ID.
*   `EMAILJS_TEMPLATE_ID`: The EmailJS template ID.
*   `EMAILJS_PUBLIC_KEY`: The EmailJS public key.

You can create a `.env` file in the root directory to define these variables.

## Dependencies

*   **Frontend:**
    *   `react`:  A JavaScript library for building user interfaces.
    *   `react-dom`:  Provides DOM-specific methods for React.
    *   `react-router-dom`:  Provides routing functionalities for React applications.
    *   `framer-motion`:  A production-ready motion library for React.
    *   `lucide-react`: A collection of beautiful icons for React.
    *   `axios`: Promise based HTTP client for the browser and node.js
    *   `vite`: Next generation frontend tooling.

*   **Backend:**
    *   `express`:  A minimal and flexible Node.js web application framework.
    *   `cors`:  Middleware for enabling Cross-Origin Resource Sharing (CORS).
    *   `@emailjs/browser`:  A library for sending emails using EmailJS.
    *   `csv-parser`:  A CSV parser for Node.js.
    *   `cheerio`:  Fast, flexible, and lean implementation of core jQuery designed specifically for the server.
    *   `node-fetch`:  A light-weight module that brings the Fetch API to Node.js
    *   `rss-parser`:  A library for parsing RSS feeds.

## Contributing

Contributions are welcome! Here's how you can contribute:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes.
4.  Test your changes thoroughly.
5.  Commit your changes with a descriptive message.
6.  Push your branch to your forked repository.
7.  Submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
