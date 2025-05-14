# Sentiment Analysis Tool

A web-based application built with Python, Flask, and NLTK to analyze the sentiment of user-provided text (e.g., customer feedback). It classifies sentiment as positive, negative, or neutral, stores the analysis for trend identification, and is designed for deployment on AWS.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [Local Installation](#local-installation)
  - [Database Configuration](#database-configuration)
- [Running the Application Locally](#running-the-application-locally)
- [Usage](#usage)
  - [Web Interface](#web-interface)
  - [API Endpoint](#api-endpoint)
- [Deployment on AWS](#deployment-on-aws)
  - [Key AWS Services](#key-aws-services)
  - [Deployment Steps (Elastic Beanstalk)](#deployment-steps-elastic-beanstalk)
- [Database for Trend Analysis](#database-for-trend-analysis)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## Features

-   **Sentiment Classification:** Classifies input text into 'positive', 'negative', or 'neutral' categories.
-   **NLTK VADER:** Utilizes NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) for robust sentiment scoring.
-   **Text Preprocessing:** Includes steps like lowercasing, punctuation removal, stop-word removal, and lemmatization for better accuracy.
-   **Web Interface:** Simple and intuitive UI built with Flask and HTML/CSS for users to input text and view results in real-time.
-   **API Endpoint:** Provides an `/analyze` API endpoint for programmatic sentiment analysis.
-   **Data Persistence:** Stores original text, processed text, sentiment label, scores, and timestamp in a database (PostgreSQL on AWS RDS, defaults to SQLite locally).
-   **Trend Analysis Ready:** Database schema designed to support querying for sentiment trends over time.
-   **AWS Deployment:** Structured for deployment on AWS (e.g., using Elastic Beanstalk and RDS).

## Project Structure

sentiment_tool/
├── app.py                 # Main Flask application, routes, DB interaction
├── sentiment_analyzer.py  # Core sentiment analysis logic (preprocessing, NLTK VADER)
├── nltk_setup.py          # Handles NLTK resource downloads
├── requirements.txt       # Python dependencies
├── .env.example           # Example for environment variables (for local PostgreSQL)
├── Procfile               # For AWS Elastic Beanstalk/Heroku (Gunicorn)
├── .ebextensions/         # AWS Elastic Beanstalk specific configurations (e.g., nltk_config.config)
│   └── nltk_config.config
├── static/                # CSS, JavaScript files
│   └── style.css
└── templates/             # HTML templates
└── index.html


## Technologies Used

-   **Backend:** Python, Flask
-   **NLP:** NLTK (Natural Language Toolkit), specifically VADER
-   **Database:** PostgreSQL (for AWS RDS), SQLite (local default)
    -   **ORM:** Flask-SQLAlchemy
-   **Frontend:** HTML, CSS, JavaScript (basic for interaction)
-   **Deployment (Target):** AWS (Elastic Beanstalk, RDS)
-   **WSGI Server:** Gunicorn

## Setup and Installation

### Prerequisites

-   Python 3.8+
-   pip (Python package installer)
-   Git
-   (Optional for local PostgreSQL) PostgreSQL server installed and running.
-   (For AWS Deployment) AWS Account, AWS CLI configured.

### Local Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/sentiment-analysis-tool.git](https://github.com/your-username/sentiment-analysis-tool.git)
    cd sentiment-analysis-tool
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK resources:**
    The application attempts to download necessary NLTK resources on startup via `nltk_setup.py`. You can also run it manually:
    ```bash
    python nltk_setup.py
    ```

### Database Configuration

The application is configured to use a PostgreSQL database when deployed or when appropriate environment variables are set. For local development, it defaults to a SQLite database (`sentiment_data.db`) if PostgreSQL environment variables are not found.

-   **For Local PostgreSQL (Optional):**
    1.  Ensure PostgreSQL is running.
    2.  Create a database and a user.
    3.  Create a `.env` file in the project root by copying `.env.example`:
        ```bash
        cp .env.example .env
        ```
    4.  Update the `.env` file with your local PostgreSQL credentials:
        ```env
        DB_USER=your_local_postgres_user
        DB_PASSWORD=your_local_postgres_password
        DB_HOST=localhost
        DB_PORT=5432
        DB_NAME=your_local_sentiment_db
        FLASK_DEBUG=True
        # SECRET_KEY=your_flask_secret_key # Add a Flask secret key for session management if needed
        ```
-   **For SQLite (Default Local):**
    No specific database configuration is needed. `app.py` will automatically create and use `sentiment_data.db` in the project root if PostgreSQL environment variables are not set.

## Running the Application Locally

1.  Ensure your virtual environment is activated and dependencies are installed.
2.  If using local PostgreSQL, ensure your `.env` file is configured and the database server is running.
3.  Start the Flask development server:
    ```bash
    python app.py
    ```
4.  Open your web browser and navigate to `http://127.0.0.1:5000/` (or the port specified if `PORT` env var is set).

## Usage

### Web Interface

Navigate to the home page (`http://127.0.0.1:5000/`). Enter the text you want to analyze in the textarea and click "Analyze Sentiment". The results (sentiment label, compound score, detailed scores) will be displayed below.

### API Endpoint

-   **Endpoint:** `/analyze`
-   **Method:** `POST`
-   **Request Body (JSON):**
    ```json
    {
        "text": "This is a wonderful product and I am very happy!"
    }
    ```
-   **Success Response (JSON):**
    ```json
    {
        "original_text": "This is a wonderful product and I am very happy!",
        "processed_text": "wonderful product happy",
        "sentiment": "positive",
        "compound_score": 0.8767,
        "detailed_scores": {
            "neg": 0.0,
            "neu": 0.293,
            "pos": 0.707,
            "compound": 0.8767
        },
        "timestamp": "YYYY-MM-DDTHH:MM:SS.ffffff"
    }
    ```
-   **Error Response (JSON):**
    ```json
    {
        "error": "No text provided"
    }
    // Status: 400
    ```

## Deployment on AWS

This application is designed to be deployed on AWS, primarily using Elastic Beanstalk for the application and RDS for the PostgreSQL database.

### Key AWS Services

-   **AWS Elastic Beanstalk:** To deploy and scale the Flask web application.
-   **AWS RDS (PostgreSQL):** As the managed relational database service.
-   **IAM:** For managing permissions.
-   **S3:** Elastic Beanstalk uses S3 to store application versions and logs.
-   **CloudWatch:** For monitoring and logging.

### Deployment Steps (Elastic Beanstalk)

1.  **Package your application:** Ensure `requirements.txt`, `Procfile`, and the `.ebextensions` directory (with `nltk_config.config`) are correctly set up.
2.  **Create an RDS PostgreSQL instance:**
    -   Configure security groups to allow inbound traffic from your Elastic Beanstalk environment's security group on port 5432.
    -   Note the DB endpoint, username, password, and DB name.
3.  **Initialize Elastic Beanstalk:**
    ```bash
    eb init -p python-3.9 <your-app-name> --region <your-region>
    ```
4.  **Create an Elastic Beanstalk environment:**
    ```bash
    eb create <your-environment-name>
    ```
5.  **Configure Environment Variables in Elastic Beanstalk:**
    Through the Elastic Beanstalk console (Configuration > Software > Environment properties), set:
    -   `DB_HOST`: Your RDS instance endpoint.
    -   `DB_NAME`: Your RDS database name.
    -   `DB_USER`: Your RDS master username.
    -   `DB_PASSWORD`: Your RDS master password.
    -   `DB_PORT`: `5432`
    -   `PYTHONPATH`: `/var/app/current` (Often helpful for module resolution)
    -   (Optionally) `FLASK_DEBUG=False` for production.
6.  **Deploy your application:**
    ```bash
    eb deploy
    ```
7.  Access your application using `eb open`.

## Database for Trend Analysis

The `SentimentRecord` table (defined in `app.py`) stores each analysis. This data can be queried to identify trends:

-   Sentiment distribution over time (daily, weekly, monthly positive/negative/neutral counts).
-   Average sentiment scores for specific periods.
-   Correlation of sentiment with events (if other data sources are integrated).

The `/trends/summary` endpoint provides a very basic example. For more advanced analysis, connect a BI tool (e.g., Amazon QuickSight) to the RDS database or build more sophisticated querying and visualization within the Flask application.

## Future Enhancements

-   [ ] More advanced text preprocessing (e.g., handling emojis, slang).
-   [ ] Train a custom sentiment model for domain-specific language.
-   [ ] User authentication and management.
-   [ ] More sophisticated trend analysis and visualization dashboards.
-   [ ] Asynchronous task processing for analysis if it becomes time-consuming.
-   [ ] Integration with other data sources for richer context.
-   [ ] Enhanced error handling and logging.
-   [ ] Unit and integration tests.

## Contributing

Contributions are welcome!

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request
