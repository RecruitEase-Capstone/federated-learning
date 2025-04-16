dir-models:
	mkdir models && mkdir models/local models/global

client-run:
	python -m client.client

server-run:
	python -m server.server	