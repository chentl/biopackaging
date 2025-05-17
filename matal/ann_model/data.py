from enum import IntFlag


class DataSource(IntFlag):
    SRC_PROJ_AT = 2 << 10
    SRC_DB = 2 << 2
    SRC_UG = 2 << 4
    SRC_OPT = 2 << 6
    SRC_ABR = 2 << 7
    SRC_TEST = 2 << 8
    SRC_RD = 2 << 9

    DA_MULTI_TESTS = 2 << 21
    DA_STD_NOISE = 2 << 22

    SRC_DA = DA_MULTI_TESTS ^ DA_STD_NOISE

    TAG_ADD_STD_NOISE = 2 << 22

    DB_UG = SRC_PROJ_AT ^ SRC_DB ^ SRC_UG

    DB = SRC_PROJ_AT ^ SRC_DB
    DB_TEST = SRC_PROJ_AT ^ SRC_DB ^ SRC_TEST

    OPT = SRC_PROJ_AT ^ SRC_OPT
    ARB = SRC_PROJ_AT ^ SRC_ABR

