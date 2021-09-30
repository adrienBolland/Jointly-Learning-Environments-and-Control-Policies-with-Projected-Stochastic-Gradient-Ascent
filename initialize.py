import agent as agent_pkg
import algo as algo_pkg
import model as model_pkg
import runner as runner_pkg
import utils
import system as system_pkg


def get_system(name, init, wrappers):
    """ instantiates the system """
    packages = utils.import_submodules(system_pkg)

    cls = utils.get_cls(name, packages)
    if cls is None:
        raise Exception(f"No system class: {name}")
    sys = cls(**init)

    for _, wrapper in wrappers.items():
        cls = utils.get_cls(wrapper["name"], packages)
        if cls is None:
            n = wrapper["name"]
            raise Exception(f"No wrapper class: {n}")
        sys = cls(sys, **wrapper["init"])

    return sys


def get_agent(sys, name, init, initialize):
    """ instantiates and initializes an agent """
    packages = utils.import_submodules(agent_pkg)

    cls = utils.get_cls(name, packages)
    if cls is None:
        raise Exception(f"No agent class: {name}")

    agent = cls(**init)
    agent.initialize(sys, **initialize)

    return agent


def get_model(sys, name, init):
    """ instantiates and initializes an agent """
    packages = utils.import_submodules(model_pkg)

    cls = utils.get_cls(name, packages)
    if cls is None:
        raise Exception(f"No model class: {name}")

    model = cls(sys, **init)

    return model


def get_logger(log_path, name, init):
    """ instantiates a logger """
    packages = utils.import_submodules(algo_pkg)

    cls = utils.get_cls(name, packages)
    if cls is None:
        raise Exception(f"No logger class: {name}")

    logger = cls(log_path, **init)

    return logger


def get_algo(sys, agent, name, init, initialize):
    """ instantiates and initializes an algorithm """
    packages = utils.import_submodules(algo_pkg)

    cls = utils.get_cls(name, packages)
    if cls is None:
        raise Exception(f"No algorithm class: {name}")

    algo = cls(sys, agent, **init)
    algo.initialize(**initialize)

    return algo


def get_runner(sys, agent, name, init):
    """ instantiates and initializes a runner """
    packages = utils.import_submodules(runner_pkg)

    cls = utils.get_cls(name, packages)
    if cls is None:
        raise Exception(f"No runner class: {name}")

    runner = cls(sys, agent, **init)

    return runner
