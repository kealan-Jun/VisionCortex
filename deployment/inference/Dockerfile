# Pass a verified existing VisionCortex model/runtime image, separate from OCR.
ARG RUNTIME_IMAGE
FROM $RUNTIME_IMAGE
USER root
WORKDIR /service
COPY . /service
RUN python -m pip install --no-deps /service
ENV HOME=/tmp VISIONCORTEX_DEFAULT_CONFIG=/service/configs/default.yaml
USER 10001:10001
CMD ["python", "-m", "uvicorn", "visioncortex.machine_api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
