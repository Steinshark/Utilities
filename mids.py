from requests_html import HTMLSession
from requests import get, post
form_url = "https://midsweb.usna.edu/ITSD/midsw/drgwq010$mids.actionquery"
cookies = {
"s_session_id" : "B369168123494312803B0F1665CD0026",
"OAMAuthnCookie_midsweb.usna.edu:443" : "sgUG29TDdIefNfJsjb9oSTbwZ8G3z5P31j%2FWXPa0JqqsuRvCOQKiMY5GffHCDen7echLxNDAHM1ofKicGQV5%2F305CbjoxP5kIPjo5snZlng8dwB8nxYsHD6DD8ecjCDNkcm1ECxVcyqwLzuhkmcJLTtVxpadrkXGbf8poAJtWYmXP9MKoNV1miEvZjdmwrhd9as36TKv1Y0hPCVe8cCEZSgHSAPH9PPoVH4NRqGLSYFL09nJ6KELVBTEGMzu16VKdMeDlq74AEs3q2L8ayuS06oPX70nKC9pQ6Wx6TOhofpTy9V%2Fpni7tfjoZ71ktxO0nIJzrImAriqzVzv9bUjmhFMDjSzsz%2BEwM%2Fz7%2BOgH0IvnCv31AnFnLuD7gTepVijlk7J39FEG2i4G5nP5SCZy%2FckOZQhvIaiTOJhjPitX13GyJvmUlj2j7H5BUK54VH0okKcyQ4MSQbEd7qRCy0vAHzXGvcd7OE518FL0TLfbyAphx57j8TMlZCNnsq%2FuFeHIKiLDzQoNF6v9QuRdDuWxTpwVtcCPmRCu85vWgTzA5nkf%2BExvd6iRCSuPtqs9hVQ3pDN2%2Fyhi1HEDLShxSo3xeA%3D%3D",
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

    file = open("22_test",'w')
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
            file.write(f"{name}:{schedule}\n")
        except IndexError:
            print(f"midn: {data['P_ALPHA']} not cooperating...")
        n += 1

if __name__ == "__main__":
    pull_Brigade(thresh=1000)
