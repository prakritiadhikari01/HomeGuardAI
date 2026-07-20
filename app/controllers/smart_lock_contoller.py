from app.application.smart_lock.smart_lock_service import (
    SmartLockVerificationService,
)


class SmartLockController:

    @staticmethod
    def verify(payload: dict):
        return SmartLockVerificationService.verify(payload)