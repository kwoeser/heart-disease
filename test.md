# Personal Website

A personal website built with modern web technologies to showcase my skills, projects, and experience.

## Overview

This repository contains the source code for my personal website. The website is designed to be a central hub for information about me, including my portfolio, blog (if applicable), and contact information. It is built using React for a dynamic and interactive user experience.

## Features

*   **Clean and Responsive Design:** The website is designed to be visually appealing and responsive across different devices (desktops, tablets, and mobile phones).
*   **Portfolio Showcase:** Highlights key projects with descriptions, screenshots, and links to live demos or GitHub repositories.
*   **Contact Form:** Allows visitors to easily send messages directly to my email.
*   **Blog (Optional):** If implemented, this section will contain articles and updates on my personal projects and interests.
*   **Dynamic Content:** React components are used to efficiently manage and update content.
*   **Route based Navigation:** Uses `react-router-dom` for navigation.
*   **Form validation:** Uses `emailjs` for form submission.

## Built With

![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoWidth=60) ![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoWidth=60) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoWidth=60) ![Node.js](https://img.shields.io/badge/Node.js-339933?style=for-the-badge&logo=node.js&logoWidth=60) ![Express.js](https://img.shields.io/badge/Express.js-000000?style=for-the-badge&logo=express&logoWidth=60)

## Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd my-portfolio
    ```

2.  **Install dependencies:**

    ```bash
    npm install
    # or
    yarn install
    # or
    pnpm install
    ```

3.  **Configure Environment Variables (if needed):**

    Some features (like the contact form or data fetching) may require environment variables. Create a `.env` file in the root directory and add the necessary variables.  Example:

    ```
    # .env
    VITE_EMAILJS_SERVICE_ID=your_service_id
    VITE_EMAILJS_TEMPLATE_ID=your_template_id
    VITE_EMAILJS_PUBLIC_KEY=your_public_key
    ```
    Make sure to use valid EmailJS keys.

4.  **Run the development server:**

    ```bash
    npm run dev
    # or
    yarn dev
    # or
    pnpm dev
    ```

    This will start the development server, and you can access the website in your browser at `http://localhost:5173/` (or the port Vite assigns).

## Usage

The website is designed to be user-friendly.  Navigate through the different sections using the navigation menu.

*   **Portfolio:** Browse my projects and view details.
*   **About:** Learn more about my background and skills.
*   **Contact:** Send me a message through the contact form.

## Contributing

Contributions are welcome! If you find a bug or have an idea for a new feature, please open an issue or submit a pull request.

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes.
4.  Test your changes thoroughly.
5.  Submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE) - see the `LICENSE` file for details.
