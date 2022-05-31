FROM registry.access.redhat.com/ubi8/python-39

COPY . .

RUN pip install --no-cache-dir -r requirements.txt
USER 0
RUN python setup.py install
USER 1001
ENTRYPOINT ["hello"]