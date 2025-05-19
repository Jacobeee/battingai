# BattingAI - MLB Batting Mechanics Analysis

A serverless application that analyzes baseball batting mechanics by comparing user videos with MLB player references.

## Overview

BattingAI is a serverless application built with AWS SAM that helps baseball players improve their batting mechanics by:

1. Analyzing user-submitted batting videos
2. Comparing them with reference videos of MLB players
3. Identifying mechanical issues and providing feedback
4. Suggesting resources for improvement

## Architecture

The application uses the following AWS services:

- **AWS Lambda**: For serverless compute
- **Amazon S3**: For storing videos and analysis results
- **Amazon API Gateway**: For handling API requests
- **AWS SAM**: For infrastructure as code

## Components

- **Video Processing**: Extracts frames from user videos
- **Video Comparison**: Compares user frames with MLB player reference frames
- **Feedback Generation**: Provides detailed feedback on batting mechanics
- **API Endpoints**: For uploading videos and retrieving results

## Prerequisites

- AWS CLI
- AWS SAM CLI
- Python 3.11
- Docker (for local testing)

## Deployment

1. Build the application:
   ```
   sam build
   ```
   
   If you encounter issues with Python not being found, you can use Docker for building:
   ```
   sam build --use-container
   ```

2. Deploy the application:
   ```
   sam deploy --guided
   ```

3. Follow the prompts to deploy the application to your AWS account.

## Usage

1. Upload a video of a person batting (preferably side view)
2. Start the analysis process
3. Retrieve the analysis results
4. Review the feedback and suggested improvements

## API Endpoints

- **POST /upload**: Upload a batting video
- **POST /analyze**: Start the analysis process
- **GET /results/{analysisId}**: Get analysis results

## Reference MLB Players

The application includes reference videos of the following MLB players:

- Bryce Harper
- Brandon Lowe

## Output Format

The analysis results are provided in JSON format with the following information:

- Mechanical issues identified
- Causes of the issues
- Corrections for the issues
- Links to resources (YouTube, Instagram) for improvement