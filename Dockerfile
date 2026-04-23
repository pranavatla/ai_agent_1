FROM public.ecr.aws/lambda/python:3.11

COPY requirements-lambda.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt

COPY app1.py ${LAMBDA_TASK_ROOT}/
COPY index.html ${LAMBDA_TASK_ROOT}/
COPY knowledge_base ${LAMBDA_TASK_ROOT}/knowledge_base

CMD ["app1.handler"]