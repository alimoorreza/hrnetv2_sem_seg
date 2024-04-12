from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import io
import os
import zipfile


def downloader(file_id, drive_service, output_file_path, output_file_dir):
    print(f"===> File ID: {file_id}")

    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    request = drive_service.files().get_media(fileId=file_id)
    fh = io.FileIO(output_file_path, 'wb')
    downloader_g = MediaIoBaseDownload(fh, request)
    done = False
    print(f"===> Downloading to {output_file_path}")

    while done is False:
        status, done = downloader_g.next_chunk()
        print("     Downloaded %d%%." % int(status.progress() * 100))

    print(f"===> Extracting {output_file_path}")
    with zipfile.ZipFile(output_file_path, "r") as zip_ref:
        zip_ref.extractall(os.path.join(output_file_dir, file_id))

    print(f"===> Extracted to {os.path.join(output_file_dir, file_id)}")
    print(f"===> Deleting {output_file_path}")
    os.remove(output_file_path)


def main(cred, output_dir, tr_dt_urls, tr_lbl_urls, tst_dt_urls, tst_lbl_urls):
    print("===> Initializing drive service....")
    credentials = service_account.Credentials.from_service_account_info(cred)
    drive_service = build('drive', 'v3', credentials=credentials)

    print("===> Initializing output directory....")
    train_data_dir = os.path.join(output_dir, 'train/images')
    train_label_dir = os.path.join(output_dir, 'train/labels')
    test_data_dir = os.path.join(output_dir, 'validation/images')
    test_label_dir = os.path.join(output_dir, 'validation/labels')

    out_dirs = [train_data_dir, train_label_dir, test_data_dir, test_label_dir]
    urls_list = [tr_dt_urls, tr_lbl_urls, tst_dt_urls, tst_lbl_urls]

    for ind_f, dir_ in enumerate(out_dirs):
        if not os.path.exists(dir_):
            os.makedirs(dir_)

        for ind_u, fl in enumerate(urls_list[ind_f]):
            output_file_path = os.path.join(dir_, f'{fl}.zip')
            downloader(fl, drive_service, output_file_path, dir_)


if __name__ == '__main__':
    output_dir_ = "/Volumes/mac_1/research/dataset/playing_for_bench_mark"

    print("===> Downloading Playing for Benchmark dataset....")
    train_data_urls = [
        "1iQg_M-y1EcXksw8zdaO4xkKT6bKauICj",
        "1nNv3WwJ81k7JoORZrNtbHXOtlxT8Txd4",
        "1R82kmXtNcX8lrT0D6kqiWda2odJNW21I",
        "1I2HK3RwGngW_XBfKUq9-e6N6gOnaOX7w",
        "1RWEyDolhCYwjWo0lNFWkGz-GtEOmxXqJ",
        "10ALTyCY0eYhin6I-EC2_JacCC7eLxLs1",
        "1kgf9pHwF-BOX6_vFFHvI-CfiaGrI2RdY",
        "1LPFS7Ay7xsewOgZRDPC23Zf_83qLmoJN",
        "1uj5lduiHYJczYGWO04nmb5feIjyti_q3",
        "1RoRbNs6N8EmeB-I7gyj50rR6hZp1WQ_y",
        "1DPh0rMyObC7kVRj0qGeoHBi3VagBAj_y",
        "1M1Ner94Mkb8F3wXGth26KbMs54QxNS35",
        "1O7gYGest2Kwd2_YgiXEaHQJXIlAq0_w5",
        "1jAfm5CMwZw5ZYCk2TpGkSpo_H6Z8VvWg",
        "1NCNPAhM-mTdpBLwjByoe67mVgZpUgHRw",
        "1NUrojiLsWWi1ppUW5zZjRO3uhbmB8_dM",
        "1SopGJ_XDDLDrvf0fTUb3IxDfiz_xNXL7",
        "1X2R-AoKglLMWNqOTi_jcz6QGKnQi5Vt9",
        "1I83Fi2joYNdBDmy_nFX3Ie-tRIQ9s80F",
        "1tuAeamXuIFi3-yLxW-M7Fl2kjzpeWMXJ",
        "18sykpkhCAbW_glVEoXoS0QNwdhQFleh2",
        "1gXl6lj_9TGsiYSkL7uINtOS3OLkih6d-",
        "1F_BPonKyUJ8902YorhMtufT0zIluySYW",
        "1dvIkrsSEX2vmaiQ24JFZCMzUybtT6Vb0",
        "1r2zrGcmL_gKIjRFuP_8S-SCwHE5If-h6",
        "1DXrOdQzFTcKJEEyHzJAV--6TJOSpeRvD",
        "1PTlG_FU3yThsBef4MHKtmov3CvXGJAz3",
        "1it0uFC3bWkpStrrRsQ124KeTy-ch_Ywr",
        "1JABMgByyneyT0qcVxDRN3m8TlHuIleeM",
        "1SJJ8c2RMEZv9OsJPJ2_cM-6ZoymDfVJF",
        "1VilHW1Yx9sRIqKzWTHzo1u43c7N-TlLQ",
        "1SfzQ5fJgi9llNXXSx9jNUlZ_wsMGMr_g",
        "1LNjjN7B7In5o7H3kH3yKPOWWBBsIa2-N",
        "1dOVfME4KzDIVoQgIpf108c8ukcc1rR3B",
        "1ewJmO6Wi0ryW-9brodh_-5KEzt5BIjRS",
        "151BEyetP58l9V5ysmT3t-Z6h--SPRi6b",
        "1--wfyLjqtmMR50SLZLwsZBjgu04q5EqL",
        "1GmggUG3qArRzY2exU9Vkd-ZxPVuVz_tY",
        "1QMHRyiSaNlhx_aSMbO7uq94KcyoINi_a",
        "1Pk5uOvqG5S3U_tmKSGQdJD4BjB0omS6q",
        "1fN6udFilsxqK6rWAGbR3EGFLQwAkrXlC",
        "1VmbRzhHM57dN7q3L-00p3-To63tWI4-y",
        "1adKhyMTDBej3FC3ZKqI3Sfc5nI0xUmJD",
        "1d-iAjDlxIhEgCrIEGenfC5VlDh-8HUFk",
        "17kTUaHQtgcUBiUMwRzuR8Z6AWuEOwZEe",
        "1LvHelLd3lyw6vKO2lmWddS2LR7w4KdJi",
        "1yf3M9YJxb_Aa1xWAluLwSsTALFK3Ffe-",
        "1zqrG7P6EOvcXkpowz9gLW8KGlXZqvIvL",
        "1foYgTg4LSq4zzm98sDMpw-mIfMgkQcLW",
        "1SPuwB4yAfjVkmg2cX3OKCemZD0yw2tI7",
        "1s5yxYPHm9xOJBQwXsRHT7B4dGtMpJwLR",
        "1p5kxaZbHQIR-9OTR73goP2VZTRAJe2hx",
        "1MdAIwOqJdpooW_exAcHsOCFDEwatYApc",
        "19is5GaDcU1mnJvLxongSHDQrkOfUfG0i",
        "1VIr46VxUgwTF5cLfRNwAmC8YCHv0afV1",
        "1HGXseRce987pAjDMFvfEYbg9QLXY9kc7",
        "1whWlXY9CtHgVrget08xTVS57GSQ3Krcd",
        "1vXmGi7rKqGX2ma0t3LWoF4hKwGmbK2jG",
        "1MFv0dcP_GEI5P3KIOefNWQjpMQ1skRYz",
        "1mUVKE97dTp5CBCb_RFwuFP1zsVpCGf3i",
        "11AGBgr0Vmd9cZbyJzRcR7X1aqGAw-G7u",
        "1KlTpNeIs1Df0kNWrEa01IRcpAta49k_c",
        "1ZmT3XLC6Y_dt52M5foylU7PUJB2M-oK2",
        "1IdtR9HAcT3KH4rvFL1cXWyxkr_soZDes",
        "1w5c6-E5CtvuNH_7I53ErzxN5IiG7LH8j",
        "1Q6lQl2fXYmeHsJxne0_5TLLQDBfwmGVP",
        "18VIVUq3MUfiQYdOjhZsZcuaIMP4o9Ilv",
        "1hyy3SzevFJpWZqz1xRcmY8sXbQf5YdpG",
        "1lxcfsHNvdNEh3vW3EDa-6VNclzWy6TY8",
        "1PhcUDAIMXR35caroFhgz8Ppl6doaRL6o",
        "1hDFjVQ6PDZdWtb087wImhEc_xYEr3oZc",
        "1oOVAp0QlCrwuOIDYoui3Q46L7FyslzN4",
        "18SsCCN8k-mbj3bHCsp8RsuCQiI1bla0d",
        "1S_7lhunMzQP8WDXLTy4IddZfq4r4MJAd",
        "1djoGdOZRFVBSautFovmyGJD-hTPu1HQa",
        "1-6MVuQMVOLoA4eWSWRfPauEUOqHBe2nl",
        "16Old81FwK5h109mF1TLTSsIJxCvPY5vf",
        "1Z4jy13OrN4p1iw4snugR5UpyDuPS1Wip",
        "1mN9lJH0jxjWzUWdqXYee-aPd25ZAIgKV",
        "1PS3qI3jQFt5dbSHph6_VqLi1pRvq7RAv",
        "1LoTPu_LuWXrVEknsGKLdoFyS9dQKeFTS",
        "155DZz7zkO8H96Aj-kZTkX_R-5lj_teY6",
        "1dOuxvLEaNC4ZUt8jh3LqU_B_BYHD6znI",
        "1Z2T7ItMSqKnIQ-jANT_ykJ7D3byVZYCU",
        "1DEk31oKdC67N7XCqOn3PShuZbJdxWURa",
        "1a-NSwU5Y6pkj6xqrShMvC_4mwi-MudXq",
        "1l_oK0xrirI9EMi4mt1ah-rd6kvjHakRn",
        "1C9Er_GMc6z-lRYbUcrIye9bjUqZhfnxb",
        "1RufM5Rz4ttBgLcY0BzXJJddkDkv885nM",
        "1Jg-lyqVYmRmP9hXZ4UMXk9dL21mfE54B",
        "1HuX1AnRybOeVMjdJ3Jri08QT9x5MF5hf",
        "1eVr0rGc1lCcfZ6ob6clcNS4h3IKMVzkO",
        "19nZVcJpq1Ekq-dryQ4HyGd1b3ybCAQVm"
    ]

    train_label_urls = [
        "1lAbmIVuQTLZu4-hNKD20wmGn1SThvFtv",
        "1KEDYhQeGQ5qOPY2RoTP1btkWmupdR2Sr",
        "1mIdQxrG_UkgHV1HBvMIjL51yaju6v0S3"
    ]

    test_data_urls = [
        "1ePxw10X_rzzuwz9altSDYITuCgjt7UTG",
        "1liZK34Dtn7pL1MkSg_KrrC6pm-DHR-KI",
        "1ui3YWKia_aRi4odcv1Y6qzmREuJjpC26",
        "13jWdvQdBhTh95Ql8quUgwNDhxQvQEMNU",
        "12DVUO53D4_-nRu2WxmJYrA8uDmgIJ-wy",
        "129WF0oWyvYQezqDLe99cPPgHv1T340WR",
        "1xc1L7s839DLSJQxk5icPVmqICGRo3AF7",
        "1x7iXZSvexT5kW1xKBklazu5MUC5oBDEt",
        "1RwdrCBsEBXG6-9EcCrqxCTuW_i0RqK1o",
        "1RgwgB4l9CezsOVfwokSlIRE_CkBzUU9w",
        "1pxLmiOueAEqITI7RiDsDRBs0vRb-ZXYB",
        "19wsDnor6O9o3TUCMDrGSUQB_i1vj3las",
        "1_PeScOq3R0Z0FbGYfP55KZ2WR6NkTeGU",
        "1O9UtdyaWJyjJSXPYgV8JhRAUYRjEnVqt",
        "1kp5ZSvCjRx8PIcOIe09ayKkLzU2x6X1w",
        "1Cqm-w6_4hjQIPFrLqtGj5YL2RMbsHhOd",
        "1LFeePKlxlzFuBWk3WobuRU9Dai-Nesvm",
        "1oIB9G1v3auclbHnVoA3swNZgXPdlzpYk",
        "1aMmlJBCsHrYGKE9Lvb8vO3TIInP07dXU",
        "1V97CHFSUx3n_i4pdvbhPmjIiwIhfLEFO",
        "1Y_3dKU8P4wofV432nw33E1eR9bo4HKsi",
        "1Wpg4fki-quGGEhrw1EfGED35YMv-aRrm",
        "1T9b5XcerwVXAQ1iEuCD0TO38Vv7gHTwZ",
        "1OBSQ2evDhXgbuu325Tf0GA3i-WZixvZr",
        "1yXCd0t9YH7U0fmb135nS7vb7eS7bjPz6",
        "1Om4n9poQY7u6gRvqSDo5XLTXr3PqVUEm",
        "1ZE-QMjBqgO4e-xP2bCSWJl1wVKoiS3IY",
        "1qrlqVrserdw5YZgzDyg0zKMs-6iQYzRA",
        "1OxdypD_ykbf3syH8rhrTGN5Em3wsNZBO",
        "1aZ9qGyw9nKx5xCPTvuzvvFXxfJmS7cD-",
        "1aAWNiQ4pXIn4YWfnyw0OcwbghY2EiH-C",
        "1lpplhVyYRUqiJthijQ1V3dTb7u5ouUpQ",
        "1Ic7qrytD9m5LFF3HcmO-ZsxcdcKbRfBc",
        "1s3WPe1Lj5EPPhRMi19WvTj7nY2w_vVgq",
        "1kQLPmFZnkStj2ZAK6BFFJj9A6PhpDGC1",
        "1-3h-MF-C0rL7Tw-xLVVU5XETtdXatS8K",
        "1ZKgW_hJHeCgZhaIPJRW-4xW7hpAdP8TK",
        "1FLq_IKSGEu2nMY2nSQuJb_lLNQkX1uop",
        "1WGSnLNo4gyP7EULhqR_-i3cxnW4M1pzK",
        "1jN9Zr85GE5BOmYSqiRHdH4eoV4NrYEyL",
        "1lHEBfIUAKNO1Z1ourrNTVv0lm9XDSp3T",
        "1jdNapQozZr0gYVSs4pmC-euVWwgfWhHP",
        "1etEqjTIwCZsVQwUrWQOUnAcijksTPjCM",
        "1Iy1pF_Te56FA0ds-QVLARMmwRn3WHVoz",
        "1_Rp50oH9sv_it1BtIW_WFqfdof_mHpE6",
        "1T9gyWoNxYMqq3ryTaYR5Z34ed_5TSF-Q",
        "1I7CjKE11NdxTl67PVZ0lgrqy01TFqgb4",
        "1WR95DQ6G5yLxmH6YKUzqLteT8sK8bcdO",
        "16VKpxfoFXckDX9sWfIVP3-tgWxcCeI_W",
        "1e8k1RB0XcAPWquAxiJmyV7_mtUQKS1Lt",
        "12XHcZjosRYeC1deLkUIk-lIqD-whwGxF",
        "1yTBpA0K1xjZ-c3aaVfAqztdxfeF-XwG0",
        "1s-V4VTUHo65ymTQFIjiU9zlDmGzFH3Ms",
        "1GhXFmxdiu82cabjE7q_6PTsZUlhm_lkW",
        "1gjQQTlk5ilhV3CH8IijBXsRd7kbeFX8V",
        "14XArvV9bIdSsVxLYCs2LBunPth0yDT7g",
        "198vxSY56nugyZJ8DDEl8HS-ej7Z4vQrF"
    ]

    test_label_urls = [
        "1QN2OSXTDsXPXntNrY-ojDpj-vFWjBZlK",
        "1XjXU9qSAvC1JB95ytv15yqyGemJqwK1U",
        "1vOgHMuRoPQ0-h-gOKoxXaVIB8ddjEbnp"
    ]

    credz = {
        "type": "service_account",
        "project_id": "eminent-maker-328215",
        "private_key_id": "c7e7d0cb59c851e73742f017bdf0948bdac673db",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQDGwFeQ0FCWpZnb\nRhJfuPqJF3ZP6LMint9rWgM29dUrAKvSEuWjpxCmQm0xoY9YP1YzY2blFEE+nMS2\n44h9nDvZbPcpUJeJ/Z0byqRuBuMDLZaBlWyM3y93IHsLGQPPSqHDr9ksmdk8BF28\nGqmN65+9MWcgzAmkSMrsoQtBJr5e4CsC2goEYYt3KdUTvNeh9TwIlDt0Zc7j7r/e\n8N6ZQ/xysTVsaGGE/Ix+6GSc73icGw83Nyl/zY9mUQJLjcAof6RklTzZcWCUkzEZ\nlyF9pkhB74cA3TTpugg2GgC67wgacBC0v5HS+0ja2egVjNA7Xp8CCfo6+o6NBAQj\nuylY9vYbAgMBAAECggEAG1Ct+DIZEUs7JP6xYcR7ckHuObdCf1yUWh2p2XDZ1iom\ncx2zv9UjHaZ9eVe56qfxwehEaPFqsh87jeMhjBnfw9cM9Pmligp4ACzOgmyh4Hrw\nd2jA3W+DB31IS9MpSeD48HsHfvC6AVycQVDswpuCfa7/fGtuW8zBTtonQNIBUeqB\nGv8Eefo/VmskvMbuzRJjWZnhzCBcF5yZMMTRIxbIX/e/gGgaBTycONxIEytdHZCs\nrWT4vP3myRo/ZJ+1J29HIG5ycz6EXzITsk0Coodyi9QyH7fZhcdLCzPGxmyFh11Q\np2Cjh46foUPt1YBzTTMSTRHq9RCNXuOSD/3c+22XmQKBgQDmNPnKOZSUiVtirIZt\nCyApf/Ll2yVRy1UeRu1hcGQvu2lLLuEyyzLAcl4jr0xQ2gJ1iPbsn0TjDPTjejlL\ncwyp/YoKp1aD7QSaOTBk0CX4ZaIvC9q4vPa31V9x8LW0lDVLIZvNm1ApwHzdYqGP\n184IA7vfKWcdUbF8XtdOMHJyVwKBgQDdBSAhEVpMxh18J3/CC2pVo5/1chTaqcpL\nv3av8/uG1R7x8DsG5Xvg3oKNkgxPUviBLbwvEJs1/5KezqB98Sb1+hWoqzakN0an\nzjgnllWtIoM70kdCoVQEZA8r9bOhAcHrYp1Nk7lbE8gw8XCkpPuAHAAOpgotsA36\nH+aHTdkn3QKBgQCSxJ2oz3MqDDSmQWZm6Rv1OWzsHP67Gk7HQeMgJ17Ii8zCsT3E\ne4Z86a2ZRK78MTI2Kz96dsSdmWocCZWzw8MEMfArgKqI67judp2i+I3swydWpVEq\nTNdjNvdmFlhSq9cacm/58xZ1uBIjtzlYEvd5ZqAF1Ei4iZgFuhW89OhSewKBgQCo\nm9xi3aYRo6GisU9ZbPP12RmSWTFnjtfw6WNV378moTc2QpoFLNRQO+9EqQITEaza\nf1VsYjLGAu6Lj/4Hlgyu4dXcEqFgrXqNUVitepZpukZ7QHN0dTZvExYv5wTd80VI\nmLaAFA33WSQDkmzobaQfmzy/8BMbv48qHWP0HINpUQKBgQCTRJiSzeCiU8wj22ph\nolwSP6qw5OinNPFbdhAC6xvMPJtNkEgVZjXmDsN1cByOj/eWjgq+lnGKXsokEhDm\nSgkL6QB98RrM5cXJ5yfZxC2n+7rdudyB4SzL8ZmgxqTeyJC/Xd1pUmkc5FS2PNlR\nQGzX6hm90Iv/Y2hjqUL4L2MbEQ==\n-----END PRIVATE KEY-----\n",
        "client_email": "downloader@eminent-maker-328215.iam.gserviceaccount.com",
        "client_id": "105783861209231134793",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/downloader%40eminent-maker-328215.iam.gserviceaccount.com"
    }

    main(credz, output_dir_, train_data_urls, train_label_urls, test_data_urls, test_label_urls)
