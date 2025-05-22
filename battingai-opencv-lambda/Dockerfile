FROM public.ecr.aws/lambda/python:3.11

# Install dependencies
RUN pip install opencv-python-headless numpy boto3

# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}

# Set the handler
CMD [ "app.lambda_handler" ]