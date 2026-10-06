# Upload and make an anonymous submission link

1. Review README.md and VALIDATION.md. Confirm the included methods and the single example match your intended
   submission package. Retain third-party
   attributions; anonymize your own identifying information.
2. Create a new empty GitHub repository, for example paper-code. Prefer private
   visibility while preparing the anonymous mirror. Do not initialize a README,
   license, or .gitignore on GitHub: this folder already has its own files.
3. Open PowerShell inside this submission folder. Initialize fresh history:

   ```powershell
   git init -b main
   git add .
   git status --short
   git diff --cached --stat
   git commit -m "Initial submission code"
   ```

   Check that .validation/, .validation-venv/, generated outputs, environments,
   and generated result directories are absent from the staged list. Run git init
   here, not in the parent development folder. Your normal Git author identity
   is fine for a private source repository; it must not appear to reviewers.
4. Copy the HTTPS URL from the new GitHub repository, replace OWNER and REPO,
   then push:

   ```powershell
   git remote add origin https://github.com/OWNER/REPO.git
   git push -u origin main
   ```

   Authenticate with your usual GitHub credential manager when prompted.
5. Open https://anonymous.4open.science/anonymize and sign in with GitHub.
   Select/paste the new repository URL and the main branch. Add your author
   names, affiliations, email addresses, account name, and repository URL to
   the redaction configuration. Keep unrelated third-party names and licenses.
   Choose an expiration covering the entire review and decision period; avoid
   redirecting to the identifying source while anonymous review is active.
   The service supports private repositories but GitHub OAuth can ask for a
   broad repository scope; review that access before granting it.
6. Open the generated anonymous link in a signed-out/private browser. Check
   the README, source, notices, and any download for identifying
   information. Test the downloadable code with the README quick example.
   Confirm the conference's current external-link policy and put the anonymous
   URL in the submission. Keep the submitted snapshot stable during review.

GitHub instructions:
https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
Anonymous GitHub service and permission details:
https://anonymous.4open.science/faq

This folder has no copied .git history. Repository creation, git init, commit,
and upload are left to the steps above.
