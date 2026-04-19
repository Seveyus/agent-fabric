from core.pipelines.project_audit.definition import PIPELINE_NAME


class PipelineRegistry:
    @staticmethod
    def exists(name: str) -> bool:
        return name == PIPELINE_NAME
