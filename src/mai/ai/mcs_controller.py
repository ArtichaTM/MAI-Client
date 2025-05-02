from typing import Callable, Generator

from mai.ai.controller import ModulesController
from mai.ai.networks.base import ModuleBase


class MCsController:
    __slots__ = ('_mc', '_target_mc')

    def __init__(
        self,
        mc: ModulesController,
        target_mc: ModulesController | None = None
    ) -> None:
        assert isinstance(mc, ModulesController)
        self._mc = mc
        self._target_mc = target_mc
        assert target_mc is None or self.__compare_assertions()

    def __iter__(self) -> Generator[ModulesController, None, None]:
        yield self._mc
        if self._target_mc:
            yield self._target_mc

    def __compare_assertions(self) -> bool:
        from mai.ai.networks.base import ModuleBase, NNModuleBase
        assert isinstance(self._target_mc, ModulesController)
        assert self._mc.training == self._target_mc.training
        for m_mc, m_t_mc in zip(
            self._mc.get_all_modules(), self._target_mc.get_all_modules()
        ):
            assert m_mc.enabled == m_t_mc.enabled, f"{m_mc}!={m_t_mc}"
            assert m_mc.loaded == m_t_mc.loaded, f"{m_mc}!={m_t_mc}"
            assert m_mc.name == m_t_mc.name, f"{m_mc}!={m_t_mc}"
            if not isinstance(m_mc, NNModuleBase):
                continue
            assert isinstance(m_t_mc, NNModuleBase)
            assert m_mc.power == m_t_mc.power, f"{m_mc}!={m_t_mc}"
            assert m_mc.file_name == m_t_mc.file_name, f"{m_mc}!={m_t_mc}"
            if m_mc.loaded:
                assert m_mc._model is not None
                assert m_t_mc._model is not None
                # Checking if parameters pointing to the same memory region
                for p1, p2 in zip(
                    m_mc._model.parameters(),
                    m_t_mc._model.parameters()
                ):
                    assert p1.data_ptr() != p2.data_ptr(), f"{p1}!={p2}"
        return True

    def _get_module(self, name: str) -> Generator['ModuleBase', None, None]:
        yield self._mc.get_module(name)
        if self._target_mc is not None:
            yield self._target_mc.get_module(name)

    def _iter_modules(self) -> Generator[
        tuple['ModuleBase', 'ModuleBase | None'],
        None, None
    ]:
        if self._target_mc is None:
            for m in self._mc.get_all_modules():
                yield m, None
            return
        for m, target_m in zip(
            self._mc.get_all_modules(),
            self._target_mc.get_all_modules()
        ):
            yield m, target_m

    def module_apply(
        self,
        name: str,
        func: Callable[['ModuleBase'], None]
    ):
        for module in self._get_module(name):
            func(module)

    def power_change(
        self,
        name: str,
        power: float
    ) -> None:
        assert isinstance(name, str)
        assert isinstance(power, float)
        assert 0. <= power <= 1.
        for module in self._get_module(name):
            module.power = power

    @property
    def training(self) -> bool:
        return self._mc.training

    @training.setter
    def training(self, value: bool) -> None:
        assert isinstance(value, bool)
        self._mc.training = value
        if self._target_mc is not None:
            self._target_mc.training = value

    def enable(self, name: str) -> None:
        assert isinstance(name, str)
        for mc in self:
            mc.module_enable(name)

    def disable(self, name: str) -> None:
        assert isinstance(name, str)
        for mc in self:
            mc.module_disable(name)

    def load(self, name: str) -> None:
        assert isinstance(name, str)
        for mc in self:
            mc.module_load(name)

    def unload(self, name: str) -> None:
        assert isinstance(name, str)
        for mc in self:
            mc.module_unload(name)

    def unload_all(self, save: bool = True) -> None:
        assert isinstance(save, bool)
        for mc in self:
            mc.unload_all_modules(save=save)

    def loaded(self) -> Generator[
        tuple['ModuleBase', 'ModuleBase | None'],
        None, None
    ]:
        for module, module_t in self._iter_modules():
            if module.loaded:
                yield module, module_t

    def enabled(self) -> Generator[
        tuple['ModuleBase', 'ModuleBase | None'],
        None, None
    ]:
        for module, module_t in self._iter_modules():
            if module.enabled:
                yield module, module_t