from models.flow_executor import FlowBatchedExecutor, FlowNonBatchedExecutor

# Dictionary mapping model types to their corresponding executor classes
executors = {
    "RealNVP": FlowBatchedExecutor,
    "RealNVP-masked": FlowBatchedExecutor,
    "RealNVP-extended": FlowBatchedExecutor,
    "tcNF-base": FlowBatchedExecutor,
    "tcNF-base-mixed": FlowBatchedExecutor,
    "tcNF-base-masked": FlowBatchedExecutor,
    "tcNF-cnn": FlowBatchedExecutor,
    "tcNF-cnn-mixed": FlowBatchedExecutor,
    "tcNF-mlp": FlowBatchedExecutor,
    "tcNF-mlp-mixed": FlowBatchedExecutor,
    "tcNF-stateless": FlowBatchedExecutor,
    "tcNF-stateless-mixed": FlowBatchedExecutor,
    "tcNF-stateful": FlowNonBatchedExecutor,
    "tcNF-stateful-mixed": FlowNonBatchedExecutor,
}


class ExecutorFactory:
    """
    Factory class to create executor instances based on the model type.
    """

    @staticmethod
    def create_executor(full_config):
        """
        Create an executor instance based on the provided model type.

        :param full_config: The full configuration dictionary.
        :return: An instance of the appropriate executor class.
        """
        executor = executors[full_config["model"]["model_type"]](full_config)
        return executor
