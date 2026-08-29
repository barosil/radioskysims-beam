"""
Repositório responsável por persistir objetos BeamParams no banco de dados.
Compatível com o modelo definido em:
    src/radioskysims/domain/beam/models.py
Inclui logging e tratamento robusto de exceções.
"""

from __future__ import annotations

import logging

from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, select

from radioskysims_beam.domain.beam.models import BeamParamsDomain
from radioskysims_core.domain.common.exceptions import BeamRepositoryError
from radioskysims_beam.infrastructure.db.beam_db_model import BeamParamsDB

# Configuração de logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BeamRepository:
    """
    Repositório SQLModel para a entidade BeamParamsDB.
    Permite salvar, recuperar, atualizar e excluir registros.

    Recursos extras:
    - Logging estruturado (info, warning, error)
    - Rollback automático em falhas
    """

    def __init__(self, session: Session):
        self.session = session

    # ----------------------------------------------------------
    # CREATE
    # ----------------------------------------------------------
    def add_beam_params(
        self, params: BeamParamsDB | BeamParamsDomain | object
    ) -> BeamParamsDB:
        """
        Adiciona um novo BeamParams no banco.
        Aceita tanto um objeto de domínio quanto um SQLModel.
        """
        try:
            if not isinstance(params, BeamParamsDB):
                logger.info(
                    "Convertendo objeto de domínio para modelo persistente BeamParamsDB"
                )
                params = BeamParamsDB.from_domain(params)

            self.session.add(params)
            self.session.commit()
            self.session.refresh(params)
            logger.info(f"BeamParams salvo com sucesso (id={params.id})")
            return params

        except SQLAlchemyError as e:
            self.session.rollback()
            logger.exception("Erro ao salvar BeamParams no banco.")
            raise BeamRepositoryError(f"Erro ao adicionar BeamParams: {e}")

    # ----------------------------------------------------------
    # READ
    # ----------------------------------------------------------
    def get_beam_params_by_id(self, beam_id: int) -> BeamParamsDB | None:
        """Retorna um BeamParams do banco pelo ID (objeto SQLModel)."""
        try:
            statement = select(BeamParamsDB).where(BeamParamsDB.id == beam_id)
            result = self.session.exec(statement).first()
            if result:
                logger.info(f"BeamParams encontrado (id={beam_id})")
            else:
                logger.warning(f"Nenhum BeamParams encontrado (id={beam_id})")
            return result
        except SQLAlchemyError as e:
            logger.exception("Erro ao buscar BeamParams por ID.")
            raise BeamRepositoryError(f"Erro ao buscar BeamParams {beam_id}: {e}")

    def list_all(self) -> list[BeamParamsDB]:
        """Retorna todos os BeamParams do banco (objetos SQLModel)."""
        try:
            statement = select(BeamParamsDB)
            results = list(self.session.exec(statement).all())
            logger.info(f"{len(results)} BeamParams encontrados no banco.")
            return results
        except SQLAlchemyError as e:
            logger.exception("Erro ao listar BeamParams.")
            raise BeamRepositoryError(f"Erro ao listar BeamParams: {e}")

    # ----------------------------------------------------------
    # UPDATE
    # ----------------------------------------------------------
    def update_beam_params(
        self,
        beam_id: int,
        new_params: BeamParamsDB | BeamParamsDomain | object,
    ) -> BeamParamsDB | None:
        """
        Atualiza um registro existente no banco.
        Pode receber tanto BeamParams persistente quanto um objeto de domínio.
        """
        try:
            existing = self.get_beam_params_by_id(beam_id)
            if not existing:
                logger.warning(
                    f"Tentativa de atualizar BeamParams inexistente (id={beam_id})"
                )
                return None

            if not isinstance(new_params, BeamParamsDB):
                new_params = BeamParamsDB.from_domain(new_params)

            for field, value in new_params.model_dump(exclude={"id"}).items():
                setattr(existing, field, value)

            self.session.add(existing)
            self.session.commit()
            self.session.refresh(existing)
            logger.info(f"BeamParams atualizado com sucesso (id={beam_id})")
            return existing

        except SQLAlchemyError as e:
            self.session.rollback()
            logger.exception("Erro ao atualizar BeamParams.")
            raise BeamRepositoryError(f"Erro ao atualizar BeamParams {beam_id}: {e}")

    # ----------------------------------------------------------
    # DELETE
    # ----------------------------------------------------------
    def delete_by_id(self, beam_id: int) -> bool:
        """Remove um BeamParams do banco pelo ID."""
        try:
            existing = self.get_beam_params_by_id(beam_id)
            if not existing:
                logger.warning(
                    f"Tentativa de deletar BeamParams inexistente (id={beam_id})"
                )
                return False

            self.session.delete(existing)
            self.session.commit()
            logger.info(f"BeamParams removido com sucesso (id={beam_id})")
            return True

        except SQLAlchemyError as e:
            self.session.rollback()
            logger.exception("Erro ao deletar BeamParams.")
            raise BeamRepositoryError(f"Erro ao deletar BeamParams {beam_id}: {e}")

    # ----------------------------------------------------------
    # DOMAIN HELPERS
    # ----------------------------------------------------------
    def get_as_domain(self, beam_id: int) -> BeamParamsDomain | None:
        """
        Retorna o objeto BeamParams convertido de volta ao domínio físico.
        """
        db_obj = self.get_beam_params_by_id(beam_id)
        if not db_obj:
            logger.warning(
                f"Tentativa de converter BeamParams inexistente (id={beam_id}) para domínio"
            )
            return None
        try:
            domain_obj = db_obj.to_domain()
            logger.info(f"BeamParams convertido para domínio (id={beam_id})")
            return domain_obj
        except Exception as e:
            logger.exception("Erro ao converter BeamParams para domínio.")
            raise BeamRepositoryError(
                f"Erro ao converter BeamParams {beam_id} para domínio: {e}"
            )

    def list_all_as_domain(self) -> list[BeamParamsDomain]:
        """
        Retorna todos os registros convertidos para o domínio.
        """
        try:
            results = [b.to_domain() for b in self.list_all()]
            logger.info(f"{len(results)} BeamParams convertidos para domínio.")
            return results
        except Exception as e:
            logger.exception("Erro ao converter lista de BeamParams para domínio.")
            raise BeamRepositoryError(f"Erro ao converter BeamParams para domínio: {e}")
