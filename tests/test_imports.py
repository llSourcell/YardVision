def test_import_cli() -> None:
    import yardvision.cli as cli  # noqa: F401


def test_import_processor() -> None:
    from vision.pipeline.processor import FrameProcessor  # noqa: F401
    from vision.core.config import AppConfig

    _ = FrameProcessor(app_config=AppConfig())

