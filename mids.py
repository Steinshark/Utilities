from requests_html import HTMLSession
from requests import get, post
from json import dumps
form_url = "https://midsweb.usna.edu/ITSD/midsw/drgwq010$mids.actionquery"
cookies = {
"s_session_id" : "B369168123494312803B0F1665CD0026",
"OAMAuthnCookie_midsweb.usna.edu:443" : "zM7Ey0J0hg73mfJw4mSF71OfqFSL4eSQige%2Fzcqmw3wuh133c74c6uQUTEjbEl0yj9oWvp6r66lZay8PL1IBI%2BtePMlBM1G%2BrCpgT%2BNCf6AXJ3wA7fv2RumXboW2aBXOLEcKfEJeEnvPn0bQQGQCBLPrS8%2FdenBOFjXoYSQftr2dXSG51haCu%2BQOyWilk0Snf%2FyuZypIrNm9IEE%2F%2BjQX1CBPLldmfAprap4odqbjfeyrnRodR9Cgj9kQ8%2Fgopzej6GbMj5cmuCKRO0UaxFJgWQtKM4QQdV05bBKho1iny%2FmY6KL8WtStJ5m%2FDygRFp5Q%2BnyXJqeGZVDtVqBrP23w8e0bhDIzoYMdg2Se8XQVRArZ%2FQEkWEZee0pXz4egfYuJwRw9UatP%2F2gdz9zMzJx2qpzQDXNPWyo2tkdSwg8Dh%2FlzOQA5FvnIFFBZX8cbk7lvYD9Z6A%2BznzKOODfMzxQj4HvLLgWUBGsnrXqlsR5Ujb8t4ROUEz0%2Fv01D7k5KhC0zBS0I8SFKA5Fxi8f9uZ80%2BBdYLvjvk13rAFNO%2BT9LTuwuS1WDzkBrDKhmYTvVtxDGhvWBnBfX%2FGSjJvJGPTzojA%3D%3D",
"JSESSIONID" : "60B15731C991F27CCBE64F20689575A5"
}


sess = HTMLSession()
from json import dumps
data = {
"P_ALPHA" : 220000,
"P_LAST_NAME" : None,
"P_MICO_CO_NBR" : None,
"P_SECOF_COOF_SEBLDA_AC_YR" : 2022,
"P_SECOF_COOF_SEBLDA_SEM" : "SPRING",
"P_SECOF_COOF_SEBLDA_BLK_NBR" : 1,
"P_MAJOR_CODE" : None,
"P_NOMI_FORMATTED_NAME" : None ,
"Z_ACTION" : "QUERY",
"Z_CHK" : 0,
}



def pull_Brigade(cl=22,begin=6, thresh=1,file='brigade'):
    mapper = {}
    n = 0
    data["P_ALPHA"] = cl*10000
    data["P_SECOF_COOF_SEBLDA_AC_YR"] = 2000 + cl

    while n/6 < thresh+150:
        data["P_ALPHA"] += 6
        try:
            resp = post(form_url,data=data,auth=('user', 'pass'),cookies=cookies)
            name = resp.text.split('<H3><TABLE ><TR><td align="_center"><font size="4" ><b><font color="#000080">')[1].split(f"/{data['P_ALPHA']}/")[0]
            schedule = resp.text.split("<P><B>Free Periods: </B>\n")[1].split('\n')[0]
            if n % 100 == 0:
                print(f"made it through {n} records")
            mapper[name] = schedule
        except IndexError:
            print(f"midn: {data['P_ALPHA']} not cooperating...")
        n += 1
    return mapper

if __name__ == "__main__":
    output = open("brigdae_22_schedules",'w')
    output.write(dumps(pull_Brigade(thresh=10)))
