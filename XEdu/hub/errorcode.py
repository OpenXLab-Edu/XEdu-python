# -*- coding: utf-8 -*-

from enum import Enum


class ErrorCode:
    def __init__(self, code, message):
        self.code = code
        self.message = message

    def __str__(self):
        return f"Error Code: {self.code}. {self.message}"

class ErrorCodeFactory:
    @staticmethod
    def error(code: int, message: str) -> ErrorCode:
        return ErrorCode(code=code, message=message)
    
    @staticmethod
    def NO_SUCH_CHECKPOINT(para: str) -> ErrorCode:
        return ErrorCode(code='-102', message=f'No such checkpoint file: {para}.')
    
    @staticmethod
    def NO_SUCH_FILE(para: str) -> ErrorCode:
        return ErrorCode(code='-103', message=f'No such file or directory: {para}.')
    
    @staticmethod
    def CHECKPOINT_TYPE(para: str) -> ErrorCode:
        return ErrorCode(code='-202', message=f'Checkpoint file type error: {para}.')

    @staticmethod
    def INFER_DATA_TYPE(para: str) -> ErrorCode:
        return ErrorCode(code='-203', message=f'Inferable file type error: {para}. Ensure your file is in one of the following formats: jpg, png, jpeg, jiff or bmp.')

    @staticmethod
    def INFER_BEFORE_SHOW() -> ErrorCode:
        return ErrorCode(code='-601', message=f'No rendered image to show. Please inference() before show().')
    
    @staticmethod
    def LOAD_CONTEXT_BEFORE_INFER() -> ErrorCode:
        return ErrorCode(code='-602', message=f'No context to inference. Please load_context() before inference().')
    
    @staticmethod
    def LABEL_MISS(para: str) -> ErrorCode:
        return ErrorCode(code='-701', message=f'Annotation for {para} is missed. Please check annotation file(.json) and relabel it.')
    

