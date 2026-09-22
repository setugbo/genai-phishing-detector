"""
Dataset Generation Module for GenAI Phishing Detector.

Generates 5000+ labeled samples across three classes:
    0 - Legitimate Financial Communication
    1 - Traditional Phishing Communication
    2 - AI-Generated Phishing Communication

Target: Nigerian financial sector communications.
"""

import csv
import json
import os
import random
from typing import List, Tuple

random.seed(42)

# ---------------------------------------------------------------------------
# Banks & financial institutions (Nigeria)
# ---------------------------------------------------------------------------
BANKS: List[str] = [
    "GTBank", "UBA", "Access Bank", "Zenith Bank", "Fidelity Bank",
    "First Bank", "Moniepoint", "PalmPay", "Opay", "Stanbic IBTC",
    "Ecobank", "Union Bank", "Wema Bank", "Sterling Bank", "Polaris Bank",
    "Keystone Bank", "FCMB", "SunTrust Bank", "Providus Bank", "TajBank",
]

FIN_TECH: List[str] = [
    "Moniepoint", "PalmPay", "Opay", "Kuda Bank", "Carbon",
    "FairMoney", "Branch", "ALAT by Wema", "VBank", "Mint",
    "Chipper Cash", "Flutterwave", "Paystack", "Interswitch",
]

ORGS: List[str] = BANKS + FIN_TECH

# ---------------------------------------------------------------------------
# Legitimate templates  (label = 0)
# ---------------------------------------------------------------------------
LEGIT_TEMPLATES: List[str] = [
    # Debit alerts
    "Dear {customer}, a debit of NGN{amount} was made on your {bank} account "
    "{account} at {merchant} on {date}. Available balance: NGN{balance}. If not you, call {phone}.",
    "Debit Alert: NGN{amount} spent at {merchant} on {date} from {bank} account "
    "{account}. Balance: NGN{balance}.",
    "Transaction alert: Withdrawal of NGN{amount} at ATM {atm_id} on {date}. "
    "{bank} account {account} balance: NGN{balance}. Thank you.",

    # Credit / transfer notifications
    "Credit Alert: NGN{amount} received from {sender} into your {bank} account "
    "{account} on {date}. Balance: NGN{balance}. Thank you for banking with us.",
    "Your transfer of NGN{amount} to {recipient} on {date} was successful. "
    "Reference: {ref}. {bank} account {account} balance: NGN{balance}.",
    "Salary payment of NGN{amount} credited to your {bank} account {account} "
    "on {date}. Balance: NGN{balance}. Regards, {bank}.",

    # Password reset / security
    "We received a password reset request for your {bank} account. "
    "Click here to reset: {link}. If you did not request this, ignore this message.",
    "Your {bank} account password was changed successfully on {date}. "
    "If you did not authorise this, contact {phone} immediately.",

    # KYC / account updates
    "Kindly update your KYC details to continue enjoying seamless banking. "
    "Visit any {bank} branch or click {link} to update. Reference: {ref}.",
    "Your BVN has been linked to your {bank} account successfully. "
    "Thank you for your cooperation.",

    # Statement / notification
    "Your monthly statement for {bank} account {account} is ready. "
    "Download at {link}. Password: your birthdate.",
    "Your {bank} account {account} has been credited with NGN{amount} "
    "being the sum of your monthly savings. Balance: NGN{balance}. Keep saving!",

    # ATM / card
    "Your {bank} debit card will expire on {date}. A new card will be "
    "delivered to your branch within 5 business days.",
    "Cardless withdrawal: Use code {ref} at any {bank} ATM to withdraw "
    "up to NGN{amount}. Valid for 1 hour.",

    # USSD / mobile
    "Dial *737# to transfer, buy airtime, or pay bills with your {bank} account.",
    "Your {bank} mobile app login was detected from a new device. "
    "If this was you, no action needed. Otherwise contact {phone}.",

    # Loan / offer (legitimate)
    "Congratulations! You are pre-qualified for a {bank} salary loan of up to "
    "NGN{amount} at {rate}% interest. Reply YES to opt in. T&C apply.",
    "Your {bank} credit card application has been approved. "
    "Your card will be delivered within 5 working days.",
]

# ---------------------------------------------------------------------------
# Traditional phishing templates  (label = 1)
# ---------------------------------------------------------------------------
TRAD_PHISH_TEMPLATES: List[str] = [
    "URGENT!!! Your BVN has been BLOCKED. Click here to verify now: {link}",
    "Dear Customer, your account will be SUSPENDED if you don't update your "
    "details now. Click: {link}",
    "Alert: NGN{amount} deducted from your account. If not you, call {phone} "
    "immediately or click {link} to reverse.",
    "Your ATM card has been deactivated. Update your PIN here: {link}",
    "Security Alert!!! Unusual login detected. Verify your account: {link}",
    "Your {bank} account requires immediate reactivation. "
    "Click here to reactivate: {link}",
    "Congratulations! You won NGN{amount} in our promotion. "
    "Claim your prize: {link}",
    "Your NIN must be linked to your BVN immediately or your account will be "
    "frozen. Verify now: {link}",
    "Dear {customer}, your internet banking has been locked. "
    "Unlock here: {link}",
    "Warning: Your account has been flagged for suspicious activity. "
    "Confirm your identity: {link}",
    "You have a pending refund of NGN{amount}. Process: {link}",
    "Your account has been credited with NGN{amount} by mistake. "
    "Return the money: {link}",
    "Dear Customer, update your account to continue enjoying our service. "
    "Click: {link}",
    "You have been selected for a loan of NGN{amount}. "
    "Accept now: {link}",
    "Your card has been charged NGN{amount} for Netflix. "
    "If not you, dispute: {link}",
    "Account upgrade available! Click to upgrade your account: {link}",
    "Your BVN has expired! Update your BVN: {link}",
    "Payment of NGN{amount} failed. Update your account: {link}",
    "Dear customer, your bank details are required for verification. "
    "Send to this email: {email}",
    "Your account will be debited NGN{amount} monthly. Cancel: {link}",
    "Immediate action required: Confirm your account details: {link}",
    "You have {count} unapproved transactions. Approve: {link}",
    "Your online banking access has been restricted. "
    "Restore access: {link}",
    "Your account has been compromised! Secure it here: {link}",
]

# ---------------------------------------------------------------------------
# AI-generated phishing templates  (label = 2)
# ---------------------------------------------------------------------------
AI_PHISH_TEMPLATES: List[str] = [
    # BVN / NIN synchronization
    "Subject: Mandatory BVN-NIN Linkage Compliance Notice\n\n"
    "Dear {customer},\n\n"
    "This is to notify you that the Central Bank of Nigeria (CBN) now requires "
    "all bank accounts to have their BVN linked to the National Identification Number (NIN) "
    "by {date}. Accounts not complying will be placed on restricted status.\n\n"
    "To complete the linkage securely, please visit: {link}\n\n"
    "This process takes less than 2 minutes.\n\n"
    "Thank you for your cooperation.\n"
    "Compliance Department\n{bank}",

    "Subject: Account Suspension Notice – Immediate Action Required\n\n"
    "Dear {customer},\n\n"
    "Our fraud detection system has identified multiple failed login attempts on your "
    "{bank} account. As a precautionary measure, your account has been temporarily "
    "suspended.\n\n"
    "To restore full access, you must verify your identity by completing the "
    "verification form at: {link}\n\n"
    "Failure to do so within 24 hours will result in permanent account closure.\n\n"
    "Regards,\n"
    "Security Operations Center\n{bank}",

    "Subject: Unauthorised Transaction Detected – Reversal Required\n\n"
    "Dear {customer},\n\n"
    "A transaction of NGN{amount} was initiated from your {bank} account "
    "to a beneficiary not on your approved list. If you did not authorise this "
    "transaction, please initiate a reversal within the next 12 hours.\n\n"
    "Reversal link: {link}\n\n"
    "Please note that failure to act may lead to permanent loss of funds.\n\n"
    "Sincerely,\n"
    "Fraud Investigation Unit\n{bank}",

    "Subject: KYC Update – Final Notice\n\n"
    "Dear {customer},\n\n"
    "This is your final reminder to update your Know Your Customer (KYC) "
    "information with {bank}. As mandated by the Central Bank of Nigeria, all "
    "accounts with incomplete KYC records will be restricted effective {date}.\n\n"
    "Update your information securely at: {link}\n\n"
    "Thank you for banking with us.\n\n"
    "Best regards,\n"
    "Customer Relationship Management\n{bank}",

    "Subject: Debit Card Replacement Notification\n\n"
    "Dear {customer},\n\n"
    "Your {bank} debit card has been flagged for replacement due to a "
    "potential security vulnerability. To avoid interruption of service, please "
    "confirm your delivery address and activate your new card using the secure "
    "portal below.\n\n"
    "Secure portal: {link}\n\n"
    "Your new card will be delivered within 3-5 business days.\n\n"
    "Warm regards,\n"
    "Card Services\n{bank}",

    "Subject: CBN Directive – Beneficiary Account Verification\n\n"
    "Dear {customer},\n\n"
    "Following the Central Bank of Nigeria's revised guidelines on electronic "
    "banking, all beneficiaries on your {bank} account must be re-verified "
    "to ensure compliance with anti-money laundering regulations.\n\n"
    "Please verify your beneficiaries at: {link}\n\n"
    "Unverified beneficiaries will be removed from your profile after {date}.\n\n"
    "Yours faithfully,\n"
    "Compliance and Risk Management\n{bank}",

    "Subject: Account Restriction Warning – Regulatory Compliance\n\n"
    "Dear {customer},\n\n"
    "Your {bank} account is scheduled for restriction due to incomplete "
    "regulatory documentation. To prevent this, please submit the required "
    "documents through our secure upload portal.\n\n"
    "Upload portal: {link}\n\n"
    "This restriction will take effect in 48 hours if not addressed.\n\n"
    "Sincerely,\n"
    "Regulatory Compliance Team\n{bank}",

    "Subject: Refund of Duplicate Charges\n\n"
    "Dear {customer},\n\n"
    "Our audit team has identified that your {bank} account was incorrectly "
    "charged NGN{amount} due to a system error. You are entitled to a full "
    "refund of this amount.\n\n"
    "To process your refund, please confirm your account details at: {link}\n\n"
    "We sincerely apologise for the inconvenience.\n\n"
    "Best regards,\n"
    "Audit and Reconciliation\n{bank}",

    "Subject: Enhanced Security Verification Required\n\n"
    "Dear {customer},\n\n"
    "As part of our ongoing commitment to protecting your financial assets, "
    "{bank} is implementing an enhanced security protocol for all online "
    "banking users.\n\n"
    "You are required to complete a multi-factor authentication setup by "
    "visiting: {link}\n\n"
    "Accounts without the enhanced security enabled will be restricted from "
    "online banking after {date}.\n\n"
    "Thank you for your understanding.\n\n"
    "Yours sincerely,\n"
    "Information Security Team\n{bank}",

    "Subject: Notification of Beneficial Ownership Declaration\n\n"
    "Dear {customer},\n\n"
    "In accordance with the Companies and Allied Matters Act (CAMA) and CBN "
    "regulations, all corporate account holders must submit a beneficial "
    "ownership declaration.\n\n"
    "File your declaration securely at: {link}\n\n"
    "Non-compliance will result in account restriction after {date}.\n\n"
    "Regards,\n"
    "Corporate Banking Division\n{bank}",

    "Subject: Pending Tax Compliance Verification\n\n"
    "Dear {customer},\n\n"
    "The Federal Inland Revenue Service (FIRS) has requested that all financial "
    "institutions verify the tax identification numbers (TIN) of their customers.\n\n"
    "Please submit your TIN for verification at: {link}\n\n"
    "Accounts with unverified TINs may be subject to withholding tax deductions "
    "at source.\n\n"
    "Thank you for your cooperation.\n\n"
    "Best regards,\n"
    "Tax Compliance Unit\n{bank}",
]

# ---------------------------------------------------------------------------
# Helper data
# ---------------------------------------------------------------------------
CUSTOMER_NAMES: List[str] = [
    "Chidi Okonkwo", "Aisha Bello", "Emeka Okafor", "Funke Adebayo",
    "Segun Ogunlade", "Ngozi Eze", "Tunde Balogun", "Fatima Usman",
    "Oluwaseun Adeyemi", "Chioma Nwosu", "Ibrahim Danjuma", "Yetunde Lawal",
    "Uchenna Obi", "Temitope Ojo", "Grace Okoro", "Kayode Adewale",
    "Halima Abubakar", "Ebuka Nwachukwu", "Ronke Adedeji", "Musa Kuti",
    "Akintunde Ogunbiyi", "Chinaza Ezeh", "Bamidele Ogun", "Folake Abiola",
    "Ifeanyi Okoro", "Kemi Alabi", "Olumide Fasanya", "Sade Bamidele",
    "Tanko Yaro", "Zainab Abdullah",
]

MERCHANTS: List[str] = [
    "ShopRite", "Jumia", "Konga", "Total Energies", "MRS Oil",
    "Shell", "SPAR", "Hubmart", "Justrite", "Addide",
    "Chicken Republic", "Mr Biggs", "Domino's Pizza", "KFC",
    "Cold Stone", "Payporte", "Slot", "Pointek", "Amazon",
    "Netflix", "Spotify", "Google Play", "Apple Store", "DStv",
    "GoTV", "Startimes", "Interswitch", "Remita", "Bet9ja", "SportyBet",
]

MONTHS: List[str] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

ACCOUNT_NUMS: List[str] = [
    "0123456789", "9876543210", "1234509876", "6789012345",
    "1112223334", "4445556667", "7778889990", "2223334445",
    "5556667778", "8889990001", "3334445556", "6667778889",
    "9990001112", "0001112223", "7890123456",
]


def _rand_amount(min_v: int = 500, max_v: int = 5_000_000) -> int:
    """Generate a random amount in Naira."""
    return random.randint(min_v, max_v)


def _rand_balance(amount: int) -> int:
    """Generate a plausible account balance > amount."""
    return amount + random.randint(1000, 50_000_000)


def _rand_date() -> str:
    day = random.randint(1, 28)
    month = random.choice(MONTHS)
    year = random.choice(["2024", "2025", "2026"])
    return f"{day}-{month}-{year}"


def _rand_bank() -> str:
    return random.choice(ORGS)


def _rand_account() -> str:
    return random.choice(ACCOUNT_NUMS)


def _rand_customer() -> str:
    return random.choice(CUSTOMER_NAMES)


def _rand_merchant() -> str:
    return random.choice(MERCHANTS)


def _rand_ref() -> str:
    return random.choice([
        "TXN" + str(random.randint(100000, 999999)),
        "REF" + str(random.randint(100000, 999999)),
        "TRN" + str(random.randint(100000, 999999)),
    ])


def _rand_link() -> str:
    domains = [
        "secure-{bank}.com", "{bank}-online.xyz", "verify-{bank}.top",
        "{bank}-login.xyz", "account-{bank}.tk", "secure-center.xyz",
        "portal-verify.com", "account-update.net", "security-check.xyz",
        "bank-verify.com",
    ]
    domain = random.choice(domains)
    # For legitimate use more realistic links
    return f"https://www.{random.choice(['gtbank.com', 'ubagroup.com', 'accessbankplc.com', 'zenithbank.com', 'fidelitybank.ng', 'firstbanknigeria.com', 'moniepoint.com', 'palmpay.com', 'opay.ng', 'kuda.com', 'carbon.co'])}/{random.choice(['reset-password', 'verify', 'statement', 'update-kyc', 'support'])}-{random.randint(1000, 9999)}"


def _rand_phone() -> str:
    prefixes = ["080", "081", "090", "070", "091"]
    return random.choice(prefixes) + "".join([str(random.randint(0, 9)) for _ in range(8)])


def _rand_email() -> str:
    return f"support@{random.choice(['bank-verify.xyz', 'account-update.com', 'secure-center.net', 'banking-portal.top'])}"


def _rand_atm_id() -> str:
    return f"ATM-{random.choice(BANKS[:5])}-{random.randint(100, 999)}"


def _generate_legitimate(amount: int) -> Tuple[str, str]:
    """Generate a legitimate financial communication. Returns (text, template_id)."""
    idx = random.randrange(len(LEGIT_TEMPLATES))
    template = LEGIT_TEMPLATES[idx]
    template_id = f"LEGIT_{idx:02d}"
    bank = _rand_bank()
    fields = {
        "customer": _rand_customer(),
        "amount": f"{amount:,}",
        "bank": bank,
        "account": _rand_account(),
        "merchant": _rand_merchant(),
        "date": _rand_date(),
        "balance": f"{_rand_balance(amount):,}",
        "phone": _rand_phone(),
        "atm_id": _rand_atm_id(),
        "ref": _rand_ref(),
        "link": _rand_link(),
        "sender": _rand_customer(),
        "recipient": _rand_customer(),
        "rate": str(random.randint(5, 25)),
        "email": _rand_email(),
        "count": str(random.randint(1, 5)),
    }
    return template.format(**fields)[:512], template_id  # cap length


def _generate_traditional_phishing(amount: int) -> Tuple[str, str]:
    """Generate a traditional phishing message. Returns (text, template_id)."""
    idx = random.randrange(len(TRAD_PHISH_TEMPLATES))
    template = TRAD_PHISH_TEMPLATES[idx]
    template_id = f"TRAD_{idx:02d}"
    bank = _rand_bank()
    shady_domains = [
        "account-verify.tk", "secure-bank.top", "update-info.xyz",
        "bank-login.ml", "verify-account.ga", "secure-center.cf",
        "portal-update.xyz", "account-reactivation.tk",
    ]
    link = f"https://{random.choice(shady_domains)}/{random.randint(10000, 99999)}"
    fields = {
        "customer": _rand_customer(),
        "amount": f"{amount:,}",
        "bank": bank,
        "link": link,
        "phone": _rand_phone(),
        "email": _rand_email(),
        "count": str(random.randint(1, 10)),
    }
    return template.format(**fields)[:512], template_id


def _generate_ai_phishing(amount: int) -> Tuple[str, str]:
    """Generate an AI-crafted phishing message. Returns (text, template_id)."""
    idx = random.randrange(len(AI_PHISH_TEMPLATES))
    template = AI_PHISH_TEMPLATES[idx]
    template_id = f"AI_{idx:02d}"
    bank = _rand_bank()
    # AI phishing uses more realistic-looking but fake URLs
    ai_domains = [
        f"secure.{bank.lower().replace(' ', '')}-portal.com",
        f"verify.{bank.lower().replace(' ', '')}-online.ng",
        f"compliance.{bank.lower().replace(' ', '')}.org",
        f"account.{bank.lower().replace(' ', '')}-secure.net",
        f"portal.{bank.lower().replace(' ', '')}-verify.com",
    ]
    link = f"https://{random.choice(ai_domains)}/{random.choice(['verify', 'compliance', 'secure', 'update', 'confirm'])}-{random.randint(1000, 9999)}"
    fields = {
        "customer": _rand_customer(),
        "amount": f"{amount:,}",
        "bank": bank,
        "link": link,
        "date": _rand_date(),
        "phone": _rand_phone(),
        "email": _rand_email(),
    }
    return template.format(**fields)[:1024], template_id  # AI phishing tends to be longer


def generate_dataset(
    n_legitimate: int = 500,
    n_trad_phish: int = 500,
    n_ai_phish: int = 500,
    output_path: str = "dataset.csv",
) -> None:
    """
    Generate the full labelled dataset and write to CSV.

    Produces two files:
      - output_path: CSV with columns [text, label] (original format)
      - output_path with _metadata suffix: CSV with [sample_id, text, label,
        template_family_id] for provenance and group-based splitting.

    Args:
        n_legitimate: Number of legitimate samples.
        n_trad_phish: Number of traditional phishing samples.
        n_ai_phish: Number of AI-generated phishing samples.
        output_path: Path where the CSV will be saved.
    """
    samples: List[Tuple[str, int, str]] = []  # (text, label, template_id)

    print(f"Generating {n_legitimate} legitimate samples ...")
    for i in range(n_legitimate):
        amount = _rand_amount(500, 500_000)  # legitimate transactions are modest
        text, tid = _generate_legitimate(amount)
        samples.append((text, 0, tid))

    print(f"Generating {n_trad_phish} traditional phishing samples ...")
    for i in range(n_trad_phish):
        amount = _rand_amount()
        text, tid = _generate_traditional_phishing(amount)
        samples.append((text, 1, tid))

    print(f"Generating {n_ai_phish} AI-generated phishing samples ...")
    for i in range(n_ai_phish):
        amount = _rand_amount()
        text, tid = _generate_ai_phishing(amount)
        samples.append((text, 2, tid))

    # Shuffle
    random.shuffle(samples)

    # Deduplicate by text
    seen: set = set()
    deduped: List[Tuple[str, int, str]] = []
    for text, label, tid in samples:
        if text not in seen:
            seen.add(text)
            deduped.append((text, label, tid))

    print(f"Total before dedup: {len(samples)} | After dedup: {len(deduped)}")

    # Write main CSV (original format: text, label)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for text, label, _tid in deduped:
            writer.writerow([text, label])

    # Write metadata CSV (sample_id, text, label, template_family_id)
    meta_path = output_path.replace(".csv", "_metadata.csv")
    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "text", "label", "template_family_id"])
        for idx, (text, label, tid) in enumerate(deduped):
            writer.writerow([f"S{idx:05d}", text, label, tid])

    # Class distribution
    from collections import Counter
    dist = Counter(label for _, label, _ in deduped)
    print(f"\nClass distribution: {dict(sorted(dist.items()))}")

    # Template family summary
    tid_dist = Counter(tid for _, _, tid in deduped)
    print(f"Unique template families: {len(tid_dist)}")
    print(f"Dataset saved to: {output_path}")
    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    generate_dataset(
        n_legitimate=1750,
        n_trad_phish=1750,
        n_ai_phish=1750,
        output_path=os.path.join(os.path.dirname(__file__), "dataset.csv"),
    )
